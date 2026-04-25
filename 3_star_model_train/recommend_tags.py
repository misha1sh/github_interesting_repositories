import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from model_tags import TagEmbeddingModel
from train_tags import build_tag_vocab, build_repo_tag_tensors

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github_token")
TAGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_3_repo_tags")
REPOS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_2_collect_api", "github_repos_100plus_stars.csv")
LLM_TAG_CACHE_PATH = os.path.join(DATA_DIR, "llm_repo_tag_cache.json")
LLM_TAG_LOG_PATH = os.path.join(DATA_DIR, "llm_repo_tag_logs.jsonl")
CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
PRINT_MIN_SCORE = 0.91


def _read_github_token() -> str:
    with open(TOKEN_PATH) as f:
        return f.read().strip()


def _github_api_get_json(url: str, token: str):
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
    )
    return r.json()


def fetch_user_stars(username: str) -> list[dict]:
    token = _read_github_token()

    all_repos = []
    page = 1
    while True:
        repos = _github_api_get_json(
            f"https://api.github.com/users/{username}/starred?per_page=100&page={page}",
            token,
        )
        if not repos or not isinstance(repos, list):
            break
        all_repos.extend(repos)
        if len(repos) < 100:
            break
        page += 1

    return [
        {
            "full_name": r["full_name"],
            "id": r["id"],
            "stargazers_count": r["stargazers_count"],
            "fork": bool(r.get("fork", False)),
        }
        for r in all_repos
    ]


def fetch_user_owned_repos(username: str) -> list[dict]:
    token = _read_github_token()

    all_repos = []
    page = 1
    while True:
        repos = _github_api_get_json(
            f"https://api.github.com/users/{username}/repos?type=owner&per_page=100&page={page}",
            token,
        )
        if not repos or not isinstance(repos, list):
            break
        all_repos.extend(repos)
        if len(repos) < 100:
            break
        page += 1

    return [
        {
            "full_name": r["full_name"],
            "fork": bool(r.get("fork", False)),
        }
        for r in all_repos
    ]


def _fetch_fork_original_full_name(repo_full_name: str, token: str) -> str | None:
    data = _github_api_get_json(f"https://api.github.com/repos/{repo_full_name}", token)
    if not isinstance(data, dict):
        return None
    source = data.get("source")
    if isinstance(source, dict) and source.get("full_name"):
        return source["full_name"]
    parent = data.get("parent")
    if isinstance(parent, dict) and parent.get("full_name"):
        return parent["full_name"]
    return None


def _load_llm_tag_cache() -> dict:
    if not os.path.exists(LLM_TAG_CACHE_PATH):
        return {}
    try:
        with open(LLM_TAG_CACHE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_llm_tag_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(LLM_TAG_CACHE_PATH), exist_ok=True)
    with open(LLM_TAG_CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_tags(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned = []
    seen = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"[^a-z0-9+._-]+", "-", item.strip().lower()).strip("-")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
        if len(cleaned) == 7:
            break
    return cleaned


def _parse_json_array_from_text(text: str):
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _fetch_repo_context(repo: str, token: str) -> str:
    meta = _github_api_get_json(f"https://api.github.com/repos/{repo}", token)
    summary = {}
    if isinstance(meta, dict):
        summary = {k: meta[k] for k in (
            "full_name", "description", "language", "topics",
            "stargazers_count", "forks_count", "size",
        ) if meta.get(k) is not None}

    readme = ""
    try:
        r = requests.get(f"https://raw.githubusercontent.com/{repo}/HEAD/README.md", timeout=10)
        if r.status_code == 200:
            readme = "\n".join(r.text.splitlines()[:150])
    except Exception:
        pass

    tree_lines = []
    subdirs = []
    try:
        contents = _github_api_get_json(f"https://api.github.com/repos/{repo}/contents", token)
        if isinstance(contents, list):
            for e in contents:
                if isinstance(e, dict):
                    name = e.get("name", "?")
                    typ = e.get("type", "?")
                    tree_lines.append(f"  {name:<40s} {typ}")
                    if typ == "dir" and not name.startswith("."):
                        subdirs.append(name)
    except Exception:
        pass

    for subdir in subdirs[:5]:
        try:
            sub = _github_api_get_json(f"https://api.github.com/repos/{repo}/contents/{subdir}", token)
            if isinstance(sub, list):
                for e in sub:
                    if isinstance(e, dict):
                        tree_lines.append(f"  {subdir}/{e.get('name','?'):<35s} {e.get('type','?')}")
        except Exception:
            pass

    return (
        "=== REPO METADATA ===\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "=== FILE TREE (root + one level) ===\n"
        f"{chr(10).join(tree_lines) or '(empty)'}\n\n"
        "=== README.md (first 150 lines) ===\n"
        f"{readme or '(empty)'}"
    )


def _fetch_file_content(repo: str, path: str) -> str:
    try:
        r = requests.get(f"https://raw.githubusercontent.com/{repo}/HEAD/{path}", timeout=10)
        if r.status_code != 200:
            return f"(HTTP {r.status_code})"
        return "\n".join(r.content.decode("utf-8", errors="replace").splitlines()[:100])
    except Exception as e:
        return f"(error: {e})"


def _extract_text_from_agent_messages(messages_iter) -> str:
    chunks = []
    for msg in messages_iter:
        for attr in ("content", "text", "result"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                chunks.append(val.strip())
            elif isinstance(val, list):
                for block in val:
                    t = getattr(block, "text", None)
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _load_claude_env() -> dict[str, str]:
    if not os.path.exists(CLAUDE_SETTINGS_PATH):
        return {}
    try:
        with open(CLAUDE_SETTINGS_PATH) as f:
            return json.load(f).get("env", {})
    except Exception:
        return {}


_sdk_options = None


def _get_sdk_options():
    global _sdk_options
    if _sdk_options is not None:
        return _sdk_options
    from claude_agent_sdk import ClaudeAgentOptions
    env = _load_claude_env()
    _sdk_options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=1,
        model="haiku",
        env=env,
    )
    return _sdk_options


async def _sdk_query(prompt: str) -> str:
    from claude_agent_sdk import query
    options = _get_sdk_options()
    chunks = []
    async for msg in query(prompt=prompt, options=options):
        for attr in ("content", "text", "result"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                chunks.append(val.strip())
            elif isinstance(val, list):
                for block in val:
                    t = getattr(block, "text", None)
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _parse_file_list(text: str) -> list[str]:
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(p).strip() for p in arr if isinstance(p, str) and p.strip()][:3]
        except Exception:
            pass
    paths = []
    for line in text.splitlines():
        line = line.strip().strip("-").strip("*").strip("`").strip()
        if not line or line.startswith("#"):
            continue
        candidate = line.split()[0] if line.split() else ""
        candidate = candidate.strip("`\"'")
        if ("/" in candidate or "." in candidate) and not candidate.startswith("http"):
            paths.append(candidate)
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))][:3]


_gh_semaphore: asyncio.Semaphore | None = None
GITHUB_API_CONCURRENCY = 3
LLM_CONCURRENCY = 4


async def _async_generate_tags(repo: str, token: str) -> tuple[str, list[str], dict]:
    global _gh_semaphore
    if _gh_semaphore is None:
        _gh_semaphore = asyncio.Semaphore(GITHUB_API_CONCURRENCY)

    t0 = time.time()

    print(f"    [{repo}] fetching context...", flush=True)
    loop = asyncio.get_event_loop()
    async with _gh_semaphore:
        context = await loop.run_in_executor(None, _fetch_repo_context, repo, token)
    print(f"    [{repo}] context fetched in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"    [{repo}] turn 1: asking LLM which files to read...", flush=True)
    try:
        file_response = await _sdk_query(
            f"Here is a GitHub repository overview:\n\n{context}\n\n"
            "Pick up to 3 source files from the file tree that best reveal what this project does.\n"
            "Skip README, LICENSE, configs, .gitignore, lockfiles.\n"
            "If the root only has directories, pick paths like dir/likely_main_file.ext.\n"
            "Respond with ONLY a JSON array of file paths, nothing else.\n"
            'Example: ["main.cpp", "src/app.py", "lib/utils.ts"]'
        )
    except BaseException as e:
        print(f"    [{repo}] turn 1 FAILED: {e}", flush=True)
        file_response = ""
    print(f"    [{repo}] turn 1 done in {time.time()-t1:.1f}s", flush=True)

    file_paths = _parse_file_list(file_response)
    print(f"    [{repo}] files to fetch: {file_paths}", flush=True)

    file_sections = ""
    async with _gh_semaphore:
        for path in file_paths:
            content = await loop.run_in_executor(None, _fetch_file_content, repo, path)
            file_sections += f"\n=== {path} (first 100 lines) ===\n{content}\n"

    t2 = time.time()
    print(f"    [{repo}] turn 2: asking LLM for tags...", flush=True)
    try:
        tag_response = await _sdk_query(
            f"Here is a GitHub repository with its source code:\n\n{context}\n"
            f"{file_sections}\n\n"
            "Based on all of the above, return ONLY a JSON array of 3-7 lowercase tags categorizing this repository.\n"
            'Example: ["python", "web", "api"]'
        )
    except BaseException as e:
        print(f"    [{repo}] turn 2 FAILED: {e}", flush=True)
        tag_response = ""
    print(f"    [{repo}] turn 2 done in {time.time()-t2:.1f}s", flush=True)

    parsed = _parse_json_array_from_text(tag_response)
    tags = _normalize_tags(parsed)
    total = time.time() - t0
    print(f"    [{repo}] total {total:.1f}s -> {tags}", flush=True)

    meta = {
        "raw_output": tag_response,
        "parsed": parsed if isinstance(parsed, list) else None,
        "files_requested": file_paths,
        "time_s": round(total, 1),
    }
    return repo, tags, meta


def enrich_profile_tags_with_llm(profile_repo_names: list[str], repo_tags: dict, as_sets: bool) -> dict:
    missing = [name for name in profile_repo_names if name not in repo_tags]
    if not missing:
        return repo_tags

    print(f"  LLM enrichment: attempting to generate tags for {len(missing)} repos without tags...")
    cache = _load_llm_tag_cache()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    print(f"  LLM logs: {LLM_TAG_LOG_PATH}")

    cache_hits = 0
    to_generate = []
    for repo_name in missing:
        cached = cache.get(repo_name)
        tags = []
        if isinstance(cached, dict):
            tags = _normalize_tags(cached.get("tags", []))
        if tags:
            cache_hits += 1
            repo_tags[repo_name] = set(tags) if as_sets else tags
            _append_jsonl(LLM_TAG_LOG_PATH, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "repo": repo_name,
                "status": "cache_hit",
                "tags": tags,
                "source": cached.get("source", "unknown"),
            })
        else:
            to_generate.append(repo_name)

    print(f"  cache hits: {cache_hits}, to generate: {len(to_generate)}")

    if not to_generate:
        return repo_tags

    try:
        from claude_agent_sdk import query as _  # noqa: F401
    except ImportError:
        print("  WARNING: claude-agent-sdk not installed. pip install claude-agent-sdk")
        return repo_tags

    token = _read_github_token()
    generated = 0
    failed = 0
    llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def _bounded_generate(repo_name: str):
        async with llm_sem:
            return await _async_generate_tags(repo_name, token)

    async def _run_all():
        nonlocal generated, failed, cache
        tasks = [asyncio.create_task(_bounded_generate(r)) for r in to_generate]
        for coro in asyncio.as_completed(tasks):
            try:
                repo_name, tags, meta = await coro
            except BaseException as e:
                print(f"    UNEXPECTED ERROR: {e}", flush=True)
                failed += 1
                continue

            log_base = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "repo": repo_name,
            }
            if tags:
                cache[repo_name] = {
                    "tags": tags,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "claude-agent-sdk",
                }
                _save_llm_tag_cache(cache)
                generated += 1
                repo_tags[repo_name] = set(tags) if as_sets else tags
                _append_jsonl(LLM_TAG_LOG_PATH, {**log_base, "status": "generated", "tags": tags, **meta})
            else:
                failed += 1
                _append_jsonl(LLM_TAG_LOG_PATH, {**log_base, "status": "failed", **meta})

    asyncio.run(_run_all())
    print(f"  LLM enrichment done: cache_hits={cache_hits}, generated={generated}, failed={failed}")
    return repo_tags


def prepare_starred_repo_names(username: str, include_forked: bool, include_user_repos: bool) -> list[str]:
    user_stars = fetch_user_stars(username)
    non_forked = [r["full_name"] for r in user_stars if not r.get("fork", False)]
    forked = [r["full_name"] for r in user_stars if r.get("fork", False)]

    starred_names = list(non_forked)
    added_originals = 0
    owned_repos = []
    owned_non_forked = []
    owned_forked = []

    if include_forked or include_user_repos:
        owned_repos = fetch_user_owned_repos(username)
        owned_non_forked = [r["full_name"] for r in owned_repos if not r.get("fork", False)]
        owned_forked = [r["full_name"] for r in owned_repos if r.get("fork", False)]

    if include_user_repos:
        starred_names.extend(owned_non_forked)
        starred_names.extend(owned_forked)

    if include_forked:
        token = _read_github_token()
        starred_names.extend(forked)
        seen = set(starred_names)
        for fork_name in set(forked + owned_forked):
            original_name = _fetch_fork_original_full_name(fork_name, token)
            if original_name and original_name not in seen:
                starred_names.append(original_name)
                seen.add(original_name)
                added_originals += 1

    print(
        f"  Found {len(user_stars)} total starred repos "
        f"({len(non_forked)} non-forked, {len(forked)} forked)"
    )
    if include_user_repos:
        print(
            f"  --include-user-repos enabled: added {len(owned_repos)} owned repos "
            f"({len(owned_non_forked)} non-forked, {len(owned_forked)} forked)"
        )
    if include_forked:
        print(
            f"  --include-forked enabled: found {len(owned_forked)} owned forked repos, "
            f"added {added_originals} fork originals"
        )
    else:
        print("  --include-forked disabled: using only non-forked starred repos")

    return list(dict.fromkeys(starred_names))


def load_tags_with_mapping(as_sets=True):
    print("Loading repo name mapping...")
    repos_df = pd.read_csv(REPOS_CSV)
    row_id_to_name = {row['repo_id']: row['repo_name'] for _, row in repos_df.iterrows()}
    print(f"  Loaded {len(row_id_to_name)} repos from CSV")
    
    print("Loading tags...")
    repo_tags = {}
    for i in range(8):
        tag_file = os.path.join(TAGS_DIR, f"tags_part_{i}.jsonl")
        if not os.path.exists(tag_file):
            continue
        with open(tag_file, "r") as f:
            for line in f:
                data = json.loads(line)
                row_id = data["repo_id"]
                if row_id in row_id_to_name:
                    repo_name = row_id_to_name[row_id]
                    repo_tags[repo_name] = set(data["tags"]) if as_sets else list(data["tags"])
    
    print(f"  Loaded tags for {len(repo_tags)} repos")
    return repo_tags


def _choose_k(embeddings: np.ndarray, max_k: int = 8) -> int:
    if embeddings.size == 0:
        return 1
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings[norms > 1e-12]
    n = int(embeddings.shape[0])
    if n < 6:
        return 1
    upper = min(max_k, n // 3, n - 1)
    if upper < 2:
        return 1
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("  WARNING: scikit-learn is not installed; falling back to a single interest cluster")
        return 1
    
    from math import log
    print(f"  n: {n}")
    min_k = max(2, int(round(log(n))))
    print(f"  min_k: {min_k}")
    
    best_k = 1
    best_score = -1.0
    for k in range(min_k, upper + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        counts = np.bincount(labels, minlength=k)
        if int(counts.min()) < 2:
            continue
        try:
            score = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _choose_k_gmm(embeddings: np.ndarray, max_k: int = 8) -> int:
    if embeddings.size == 0:
        return 1
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings[norms > 1e-12]
    n = int(embeddings.shape[0])
    if n < 6:
        return 1
    upper = min(max_k, n // 2, n - 1)
    if upper < 2:
        return 1
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("  WARNING: scikit-learn is not installed; falling back to a single interest cluster")
        return 1

    # For recommendation diversification we prefer at least two clusters when feasible.
    from math import log
    min_k = max(2, int(round(log(n))))
    best_k = min_k
    best_obj = None
    for k in range(min_k, upper + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                random_state=42,
                n_init=20,
                reg_covar=1e-5,
                covariance_type="full",
                init_params="kmeans",
            )
            labels = gmm.fit_predict(embeddings)
            counts = np.bincount(labels, minlength=k)
            if int(counts.min()) < 2:
                continue
            # AIC penalizes complexity less aggressively than BIC, which helps
            # avoid over-collapsing to one component on small user profiles.
            aic = float(gmm.aic(embeddings))
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            continue
        # Multi-objective: prefer lower AIC and better separation.
        obj = -sil # (aic, -sil)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_k = k

    if best_obj is None:
        return 2

    return best_k


def _cluster_starred_embeddings(embeddings: np.ndarray, clustering_type: str, chosen_k: int):
    n = int(embeddings.shape[0])
    if n <= 1 or chosen_k <= 1:
        return [0] * n
    norms = np.linalg.norm(embeddings, axis=1)
    nonzero_idx = np.where(norms > 1e-12)[0]
    if nonzero_idx.size == 0:
        return [-1] * n
    if nonzero_idx.size <= 1 or chosen_k <= 1:
        labels = np.full(n, -1, dtype=np.int32)
        labels[nonzero_idx] = 0
        return labels.tolist()
    emb_nz = embeddings[nonzero_idx]
    chosen_k = min(chosen_k, int(nonzero_idx.size))
    try:
        from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("  WARNING: scikit-learn is not installed; using a single cluster")
        return [0] * n

    if clustering_type == "k-means":
        km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
        labels_nz = km.fit_predict(emb_nz)
        labels = np.full(n, -1, dtype=np.int32)
        labels[nonzero_idx] = labels_nz
        return labels.tolist()

    if clustering_type == "agglomerative":
        try:
            agg = AgglomerativeClustering(n_clusters=chosen_k, metric="cosine", linkage="average")
        except TypeError:
            agg = AgglomerativeClustering(n_clusters=chosen_k, affinity="cosine", linkage="average")
        labels_nz = agg.fit_predict(emb_nz)
        labels = np.full(n, -1, dtype=np.int32)
        labels[nonzero_idx] = labels_nz
        return labels.tolist()

    if clustering_type == "gmm":
        labels_nz = None
        for k in range(chosen_k, 1, -1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=42,
                    n_init=20,
                    reg_covar=1e-5,
                    covariance_type="full",
                    init_params="kmeans",
                )
                cand = gmm.fit_predict(emb_nz)
                counts = np.bincount(cand, minlength=k)
                if int(counts.min()) < 2:
                    continue
                labels_nz = cand
                break
            except Exception:
                continue
        if labels_nz is None:
            labels_nz = np.zeros(len(emb_nz), dtype=np.int32)
        labels = np.full(n, -1, dtype=np.int32)
        labels[nonzero_idx] = labels_nz
        return labels.tolist()

    if clustering_type == "dbscan":
        target_k = max(2, chosen_k)
        best_labels = None
        best_score = None
        eps_values = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
        min_samples_values = sorted(set([2, 3, max(2, n // 20)]))
        for eps in eps_values:
            for min_samples in min_samples_values:
                labels_nz = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb_nz)
                cluster_count = len(set(labels_nz.tolist()) - {-1})
                if cluster_count <= 0:
                    continue
                non_noise = int((labels_nz != -1).sum())
                score = (-abs(cluster_count - target_k), non_noise)
                if best_score is None or score > best_score:
                    best_score = score
                    best_labels = labels_nz
        if best_labels is None:
            print("  WARNING: DBSCAN found no clusters; using a single cluster")
            labels = np.full(n, -1, dtype=np.int32)
            labels[nonzero_idx] = 0
            return labels.tolist()
        labels = np.full(n, -1, dtype=np.int32)
        labels[nonzero_idx] = best_labels
        return labels.tolist()

    print(f"  WARNING: Unknown clustering type '{clustering_type}', using k-means")
    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    labels_nz = km.fit_predict(emb_nz)
    labels = np.full(n, -1, dtype=np.int32)
    labels[nonzero_idx] = labels_nz
    return labels.tolist()


def recommend(username: str, top_k: int = 30, include_forked: bool = False, include_user_repos: bool = False):
    print(f"Fetching starred repos for {username}...")
    starred_names = prepare_starred_repo_names(username, include_forked, include_user_repos)
    print(f"  Using {len(starred_names)} repos for profile building")
    for name in starred_names[:10]:
        print(f"    {name}")
    if len(starred_names) > 10:
        print(f"    ... and {len(starred_names) - 10} more")

    print()
    repo_tags = load_tags_with_mapping()
    if include_user_repos:
        repo_tags = enrich_profile_tags_with_llm(starred_names, repo_tags, as_sets=True)

    print("\nLoading graph metadata...")
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
    repo_names = meta["repo_names"]

    print("\nBuilding user tag profile from starred repos...")
    user_tag_counts = defaultdict(float)
    known_starred_names = []
    unknown_starred = []
    
    for name in starred_names:
        if name in repo_tags:
            known_starred_names.append(name)
            for tag in repo_tags[name]:
                user_tag_counts[tag] += 1.0
        else:
            unknown_starred.append(name)
    
    print(f"  {len(known_starred_names)} starred repos have tags, {len(unknown_starred)} don't")
    
    if not user_tag_counts:
        print("ERROR: No tags found for starred repos!")
        return
    
    total_weight = sum(user_tag_counts.values())
    print(f"  User tag profile: {len(user_tag_counts)} unique tags")
    
    sorted_tags = sorted(user_tag_counts.items(), key=lambda x: -x[1])
    print(f"  Top tags: {', '.join([f'{tag}({int(count)})' for tag, count in sorted_tags[:15]])}")

    print("\nScoring all repos in graph...")
    scores = []
    repo_indices = []
    
    starred_set = set(starred_names)
    
    for idx, name in enumerate(repo_names):
        if name in starred_set:
            continue
        
        if name not in repo_tags:
            continue
        
        score = 0.0
        for tag in repo_tags[name]:
            if tag in user_tag_counts:
                score += user_tag_counts[tag] / total_weight
        
        if score > 0:
            scores.append(score)
            repo_indices.append(idx)
    
    print(f"  Scored {len(scores)} candidate repos")
    
    if len(scores) < top_k:
        print(f"WARNING: Only {len(scores)} candidates found, less than top_k={top_k}")
        top_k = len(scores)
    
    if not scores:
        print("ERROR: No candidate repos found!")
        return
    
    scores = np.array(scores, dtype=np.float32)
    repo_indices = np.array(repo_indices, dtype=np.int32)
    
    topk_mask = np.argpartition(scores, -top_k)[-top_k:]
    topk_mask = topk_mask[np.argsort(-scores[topk_mask])]
    
    topk_indices = repo_indices[topk_mask]
    topk_scores = scores[topk_mask]

    print(f"\n{'='*80}")
    print(f"Top {top_k} Recommended Repositories for @{username} (Tag-Based Algorithm)")
    print(f"{'='*80}")
    for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores)):
        if float(score) < PRINT_MIN_SCORE:
            continue
        name = repo_names[idx]
        tags = repo_tags.get(name, set())
        matching_tags = sorted([tag for tag in tags if tag in user_tag_counts], key=lambda t: -user_tag_counts[t])
        print(f"  {rank+1:2d}. {name:<50s}  (score: {score:.4f})")
        if matching_tags:
            print(f"      Matching tags: {', '.join(matching_tags[:10])}")


def recommend_nn(
    username: str,
    top_k: int = 30,
    device: str = "cuda",
    include_forked: bool = False,
    include_user_repos: bool = False,
    clusters: int = 0,
    clustering_type: str = "k-means",
):
    print(f"Fetching starred repos for {username}...")
    starred_names = prepare_starred_repo_names(username, include_forked, include_user_repos)
    print(f"  Using {len(starred_names)} repos for profile building")
    for name in starred_names[:10]:
        print(f"    {name}")
    if len(starred_names) > 10:
        print(f"    ... and {len(starred_names) - 10} more")

    repo_tags_original = load_tags_with_mapping(as_sets=False)

    print("\nLoading graph metadata...")
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
    repo_names = meta["repo_names"]

    print("Building tag vocabulary and repo tensors (original tags only)...")
    tag2idx = build_tag_vocab(repo_tags_original)
    tag_indices, tag_weights = build_repo_tag_tensors(repo_names, repo_tags_original, tag2idx)
    tag_indices = tag_indices.to(device)
    tag_weights = tag_weights.to(device)
    print(f"  {len(tag2idx)} unique tags")

    print("Loading trained model...")
    ckpt_path = os.path.join(DATA_DIR, "best_model_tags.pt")
    state = torch.load(ckpt_path, weights_only=True)
    embed_dim = state[[k for k in state if "tag_embeddings" in k][0]].shape[1]
    hidden_key = next((k for k in state if k.endswith("net.0.weight") or k.endswith("_orig_mod.net.0.weight")), None)
    hidden_dim = state[hidden_key].shape[0] if hidden_key is not None else 0

    model = TagEmbeddingModel(num_tags=len(tag2idx), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    fixed = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()

    print("Computing all repo embeddings...")
    with torch.no_grad():
        repo_embs = model(tag_indices, tag_weights)  # [num_repos, embed_dim]

    if include_user_repos:
        repo_tags_original = enrich_profile_tags_with_llm(starred_names, repo_tags_original, as_sets=False)

    print("\nBuilding user embedding from starred repos...")
    starred_set = set(starred_names)
    known = []
    unknown = []
    name_to_idx = {name: idx for idx, name in enumerate(repo_names)}
    for name in starred_names:
        if name in name_to_idx:
            known.append(name_to_idx[name])
        else:
            unknown.append(name)

    print(f"  {len(known)} starred repos found in graph, {len(unknown)} not found")

    extra_embs = None
    unknown_with_tags = []
    if unknown:
        unknown_with_tags = [n for n in unknown if n in repo_tags_original and repo_tags_original[n]]
        if unknown_with_tags:
            extra_ti, extra_tw = build_repo_tag_tensors(unknown_with_tags, repo_tags_original, tag2idx)
            extra_ti = extra_ti.to(device)
            extra_tw = extra_tw.to(device)
            with torch.no_grad():
                extra_embs = model(extra_ti, extra_tw)
            print(f"  {len(unknown_with_tags)} unknown repos have tags -> computed embeddings on-the-fly")

    if not known and extra_embs is None:
        print("ERROR: No starred repos found in graph and no tags for unknown repos!")
        return

    parts = []
    starred_repo_names_ordered = []
    if known:
        parts.append(repo_embs[torch.tensor(known, device=device)])
        starred_repo_names_ordered.extend([repo_names[idx] for idx in known])
    if extra_embs is not None:
        parts.append(extra_embs)
        starred_repo_names_ordered.extend(unknown_with_tags)
    starred_embs = torch.cat(parts, dim=0)
    starred_norms = torch.linalg.vector_norm(starred_embs, dim=1)
    zero_starred = int((starred_norms <= 1e-12).sum().item())
    if zero_starred > 0:
        print(
            f"  WARNING: {zero_starred}/{int(starred_embs.shape[0])} starred repo embeddings are zero "
            "(usually tags missing from trained vocab); excluding them from profile"
        )
    keep_mask = starred_norms > 1e-12
    if int(keep_mask.sum().item()) == 0:
        print("ERROR: All starred repos have zero effective tags in trained vocabulary.")
        return
    if zero_starred > 0:
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()
        starred_embs = starred_embs[keep_mask]
        starred_repo_names_ordered = [starred_repo_names_ordered[i] for i in keep_idx]
    n_starred = int(starred_embs.shape[0])
    if clusters < 0:
        print("  WARNING: --clusters cannot be negative; using auto mode")
        clusters = 0
    if clusters == 1 or n_starred < 6:
        chosen_k = 1
    elif clusters > 1:
        chosen_k = min(clusters, n_starred)
    else:
        emb_np = starred_embs.detach().cpu().numpy()
        if clustering_type == "gmm":
            chosen_k = _choose_k_gmm(emb_np, max_k=8)
        else:
            chosen_k = _choose_k(emb_np, max_k=8)
    print(f"  Clustering type: {clustering_type}")
    print(f"  Clustering starred interests into {chosen_k} cluster(s)")

    repo_tags_sets = {name: set(tags) for name, tags in repo_tags_original.items()}
    user_tag_counts = defaultdict(float)
    for name in set(starred_repo_names_ordered):
        for tag in repo_tags_sets.get(name, set()):
            user_tag_counts[tag] += 1.0

    norm_repo_embs = F.normalize(repo_embs, dim=-1)
    excluded_idx = set(known)
    valid_mask_base = torch.ones(norm_repo_embs.shape[0], dtype=torch.bool, device=device)
    if excluded_idx:
        valid_mask_base[torch.tensor(sorted(excluded_idx), device=device)] = False
    valid_count = int(valid_mask_base.sum().item())
    if valid_count == 0:
        print("ERROR: No candidate repos left after filtering.")
        return
    if top_k > valid_count:
        top_k = valid_count

    if chosen_k == 1:
        cluster_ids = [0] * n_starred
    else:
        cluster_ids = _cluster_starred_embeddings(
            starred_embs.detach().cpu().numpy(),
            clustering_type=clustering_type,
            chosen_k=chosen_k,
        )

    cluster_members = defaultdict(list)
    cluster_member_idx = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if int(cid) < 0:
            continue
        cluster_members[int(cid)].append(starred_repo_names_ordered[i])
        cluster_member_idx[int(cid)].append(i)
    cluster_order = sorted(cluster_members.keys(), key=lambda c: -len(cluster_members[c]))
    if not cluster_order:
        cluster_order = [0]
        cluster_members = defaultdict(list, {0: list(starred_repo_names_ordered)})
        cluster_member_idx = defaultdict(list, {0: list(range(n_starred))})

    centroids = []
    for cid in cluster_order:
        member_idx = cluster_member_idx[cid]
        if not member_idx:
            centroids.append(F.normalize(starred_embs.mean(dim=0), dim=-1))
            continue
        centroids.append(F.normalize(starred_embs[member_idx].mean(dim=0), dim=-1))
    cluster_count = len(cluster_order)

    centroids_tensor = torch.stack(centroids, dim=0)
    cluster_scores_matrix = centroids_tensor @ norm_repo_embs.t()
    if excluded_idx:
        cluster_scores_matrix[:, torch.tensor(sorted(excluded_idx), device=device)] = -1e9

    base = top_k // max(1, cluster_count)
    rem = top_k % max(1, cluster_count)
    cluster_quota = {}
    for rank, cid in enumerate(cluster_order):
        cluster_quota[cid] = base + (1 if rank < rem else 0)

    print(f"\n{'='*80}")
    print(f"Top {top_k} Recommended Repositories for @{username} (Tag NN Model, Clustered)")
    print(f"{'='*80}")
    recommended_global = set()
    for c_rank, cid in enumerate(cluster_order, start=1):
        member_names = cluster_members[cid]
        tag_counts = defaultdict(float)
        for name in member_names:
            for tag in repo_tags_sets.get(name, set()):
                tag_counts[tag] += 1.0
        top_cluster_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])[:5]]
        tags_str = ", ".join(top_cluster_tags) if top_cluster_tags else "(no tags)"
        print(f"\n[Cluster {c_rank}/{cluster_count}] members={len(member_names)} top-tags: {tags_str}")

        quota = cluster_quota.get(cid, 0)
        if quota <= 0:
            continue
        scores = cluster_scores_matrix[c_rank - 1].clone()
        if recommended_global:
            scores[torch.tensor(sorted(recommended_global), device=device)] = -1e9
        valid_scores = scores[scores > -1e8]
        if valid_scores.numel() == 0:
            print("  No candidates left after deduplication.")
            continue
        take = min(quota, int(valid_scores.numel()))
        topk_scores, topk_indices = scores.topk(take)
        sorted_valid = torch.sort(valid_scores).values
        valid_mean = valid_scores.mean()
        valid_std = valid_scores.std(unbiased=False).clamp(min=1e-8)

        for rank, (idx, score) in enumerate(zip(topk_indices.tolist(), topk_scores.tolist()), start=1):
            if float(score) < PRINT_MIN_SCORE:
                continue
            recommended_global.add(idx)
            name = repo_names[idx]
            tags = repo_tags_sets.get(name, set())
            matching = sorted([t for t in tags if t in tag_counts], key=lambda t: -tag_counts[t])
            score_t = torch.tensor(score, device=device)
            z = (score_t - valid_mean) / valid_std
            calibrated = torch.sigmoid(z).item()
            pct_rank = torch.searchsorted(sorted_valid, score_t, right=True).float() / float(sorted_valid.numel())
            print(f"  {rank:2d}. {name:<50s}  (score: {score:.4f}, cal: {calibrated:.4f}, pct: {pct_rank.item():.4f})")
            if matching:
                print(f"      Tags: {', '.join(matching[:10])}")

    user_emb_overall = F.normalize(starred_embs.sum(dim=0, keepdim=True), dim=-1)
    overall_scores = (user_emb_overall @ norm_repo_embs.t()).squeeze(0)
    if excluded_idx:
        overall_scores[torch.tensor(sorted(excluded_idx), device=device)] = -1e9
    valid_scores = overall_scores[overall_scores > -1e8]
    if valid_scores.numel() == 0:
        return
    overall_k = min(top_k, int(valid_scores.numel()))
    topk_scores, topk_indices = overall_scores.topk(overall_k)
    sorted_valid = torch.sort(valid_scores).values
    valid_mean = valid_scores.mean()
    valid_std = valid_scores.std(unbiased=False).clamp(min=1e-8)
    score_quantiles = torch.quantile(valid_scores, torch.tensor([0.5, 0.9, 0.99], device=device))
    topk_std = topk_scores.std(unbiased=False).item() if topk_scores.numel() > 1 else 0.0
    top1_top10_gap = (topk_scores[0] - topk_scores[min(9, topk_scores.numel() - 1)]).item()

    print(f"\n{'='*80}")
    print(f"Overall Top {overall_k} (global summed profile, no clusters)")
    print(f"{'='*80}")
    print(
        f"Score spread | p50/p90/p99: {score_quantiles[0].item():.4f}/{score_quantiles[1].item():.4f}/{score_quantiles[2].item():.4f}"
        f" | top-k std: {topk_std:.6f} | top1-top10 gap: {top1_top10_gap:.6f}"
    )
    for rank, (idx, score) in enumerate(zip(topk_indices.tolist(), topk_scores.tolist()), start=1):
        if float(score) < PRINT_MIN_SCORE:
            continue
        name = repo_names[idx]
        tags = repo_tags_sets.get(name, set())
        matching = sorted([t for t in tags if t in user_tag_counts], key=lambda t: -user_tag_counts[t])
        score_t = torch.tensor(score, device=device)
        z = (score_t - valid_mean) / valid_std
        calibrated = torch.sigmoid(z).item()
        pct_rank = torch.searchsorted(sorted_valid, score_t, right=True).float() / float(sorted_valid.numel())
        print(f"  {rank:2d}. {name:<50s}  (score: {score:.4f}, cal: {calibrated:.4f}, pct: {pct_rank.item():.4f})")
        if matching:
            print(f"      Tags: {', '.join(matching[:10])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--model", type=str, default="nn", choices=["simple", "nn"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-forked", action="store_true")
    parser.add_argument("--include-user-repos", action="store_true")
    parser.add_argument("--clusters", type=int, default=0, help="Number of interest clusters for --model nn (0=auto)")
    parser.add_argument(
        "--clustering-type",
        type=str,
        default="k-means",
        choices=["k-means", "agglomerative", "gmm", "dbscan"],
        help="Clustering algorithm for --model nn",
    )
    args = parser.parse_args()
    if args.model == "nn":
        recommend_nn(
            args.username,
            args.top_k,
            args.device,
            args.include_forked,
            args.include_user_repos,
            args.clusters,
            args.clustering_type,
        )
    else:
        recommend(args.username, args.top_k, args.include_forked, args.include_user_repos)


if __name__ == "__main__":
    main()
