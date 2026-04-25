import os
import sys
import json
import re
import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Optional

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # 4_website/backend
_WEBSITE_DIR = os.path.dirname(_BACKEND_DIR)                # 4_website
_REPO_DIR = os.path.dirname(_WEBSITE_DIR)                   # github (repo root)
_TRAIN_DIR = os.path.join(_REPO_DIR, "3_star_model_train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from model_tags import TagEmbeddingModel
from train_tags import build_tag_vocab, build_repo_tag_tensors
from recommend_tags import enrich_profile_tags_with_llm

DATA_DIR = os.path.join(_TRAIN_DIR, "processed")
TOKEN_PATH = os.path.join(_REPO_DIR, ".github_token")
TAGS_DIR = os.path.join(_REPO_DIR, "1_3_repo_tags")
REPOS_CSV = os.path.join(_REPO_DIR, "1_2_collect_api", "github_repos_100plus_stars.csv")
LLM_TAG_CACHE_PATH = os.path.join(DATA_DIR, "llm_repo_tag_cache.json")
LLM_TAG_LOG_PATH = os.path.join(DATA_DIR, "llm_repo_tag_logs.jsonl")
CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")

VALID_MODELS = {"nn", "simple"}
VALID_CLUSTERING = {"k-means", "agglomerative", "gmm", "dbscan"}
VALID_DEVICES = {"cuda", "cpu"}


def detect_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def validate_params(params: dict) -> list[str]:
    errors = []
    username = params.get("username", "")
    if not username or not isinstance(username, str) or not username.strip():
        errors.append("username is required")
    elif not re.match(r"^[A-Za-z0-9_.-]{1,39}$", username.strip()):
        errors.append("username contains invalid characters")

    top_k = params.get("top_k", 30)
    if not isinstance(top_k, int) or not (1 <= top_k <= 200):
        errors.append("top_k must be an integer between 1 and 200")

    model_type = params.get("model", "nn")
    if model_type not in VALID_MODELS:
        errors.append(f"model must be one of {sorted(VALID_MODELS)}")

    clusters = params.get("clusters", 0)
    if not isinstance(clusters, int) or clusters < 0:
        errors.append("clusters must be a non-negative integer")

    clustering_type = params.get("clustering_type", "k-means")
    if clustering_type not in VALID_CLUSTERING:
        errors.append(f"clustering_type must be one of {sorted(VALID_CLUSTERING)}")

    return errors


class StaticData:
    """Holds preloaded static data shared across requests."""
    repo_names: list = []
    repo_tags_original: dict = {}
    repo_tags_sets: dict = {}
    tag2idx: dict = {}
    tag_indices: Optional[torch.Tensor] = None
    tag_weights: Optional[torch.Tensor] = None
    model: Optional[TagEmbeddingModel] = None
    repo_embs: Optional[torch.Tensor] = None
    norm_repo_embs: Optional[torch.Tensor] = None
    loaded: bool = False
    load_error: Optional[str] = None


_static: StaticData = StaticData()


def preload(emit: Callable = None):
    """Preload tags, model and embeddings into memory. Called once at startup."""
    if emit is None:
        emit = lambda msg, **kw: None

    if _static.loaded:
        emit("Static data already loaded.", event="progress")
        return

    try:
        emit("Loading repo name mapping...", event="progress")
        repos_df = pd.read_csv(REPOS_CSV)
        row_id_to_name = {row["repo_id"]: row["repo_name"] for _, row in repos_df.iterrows()}
        emit(f"  Loaded {len(row_id_to_name)} repos from CSV", event="progress")

        emit("Loading tags...", event="progress")
        repo_tags = {}
        for i in range(8):
            tag_file = os.path.join(TAGS_DIR, f"tags_part_{i}.jsonl")
            if not os.path.exists(tag_file):
                continue
            with open(tag_file) as f:
                for line in f:
                    data = json.loads(line)
                    row_id = data["repo_id"]
                    if row_id in row_id_to_name:
                        repo_tags[row_id_to_name[row_id]] = list(data["tags"])
        emit(f"  Loaded tags for {len(repo_tags)} repos", event="progress")

        emit("Loading graph metadata...", event="progress")
        meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
        repo_names = meta["repo_names"]
        emit(f"  {len(repo_names)} repos in graph", event="progress")

        emit("Building tag vocabulary...", event="progress")
        tag2idx = build_tag_vocab(repo_tags)
        emit(f"  {len(tag2idx)} unique tags", event="progress")

        emit("Building repo tag tensors...", event="progress")
        tag_indices, tag_weights = build_repo_tag_tensors(repo_names, repo_tags, tag2idx)
        emit("  Tensors built", event="progress")

        emit("Loading trained model...", event="progress")
        ckpt_path = os.path.join(DATA_DIR, "best_model_tags.pt")
        state = torch.load(ckpt_path, weights_only=True)
        embed_dim = state[[k for k in state if "tag_embeddings" in k][0]].shape[1]
        hidden_key = next(
            (k for k in state if k.endswith("net.0.weight") or k.endswith("_orig_mod.net.0.weight")), None
        )
        hidden_dim = state[hidden_key].shape[0] if hidden_key else 0
        model = TagEmbeddingModel(num_tags=len(tag2idx), embed_dim=embed_dim, hidden_dim=hidden_dim)
        fixed = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(fixed, strict=False)
        model.eval()
        emit(f"  Model loaded (embed_dim={embed_dim}, hidden_dim={hidden_dim})", event="progress")

        _static.repo_names = repo_names
        _static.repo_tags_original = repo_tags
        _static.repo_tags_sets = {n: set(t) for n, t in repo_tags.items()}
        _static.tag2idx = tag2idx
        _static.tag_indices = tag_indices
        _static.tag_weights = tag_weights
        _static.model = model
        _static.loaded = True
        emit("Preloading complete.", event="progress")

    except Exception as e:
        _static.load_error = str(e)
        emit(f"Preload error: {e}", event="error")
        raise


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


def fetch_user_stars(username: str, emit: Callable) -> list[dict]:
    token = _read_github_token()
    all_repos = []
    page = 1
    while True:
        repos = _github_api_get_json(
            f"https://api.github.com/users/{username}/starred?per_page=100&page={page}", token
        )
        if not repos or not isinstance(repos, list):
            break
        all_repos.extend(repos)
        emit(f"  Fetched page {page} ({len(repos)} repos)...", event="progress")
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
            f"https://api.github.com/users/{username}/repos?type=owner&per_page=100&page={page}", token
        )
        if not repos or not isinstance(repos, list):
            break
        all_repos.extend(repos)
        if len(repos) < 100:
            break
        page += 1
    return [{"full_name": r["full_name"], "fork": bool(r.get("fork", False))} for r in all_repos]


def _fetch_fork_original(repo_full_name: str, token: str) -> Optional[str]:
    data = _github_api_get_json(f"https://api.github.com/repos/{repo_full_name}", token)
    if not isinstance(data, dict):
        return None
    for key in ("source", "parent"):
        src = data.get(key)
        if isinstance(src, dict) and src.get("full_name"):
            return src["full_name"]
    return None


def prepare_starred_repo_names(username: str, include_forked: bool, include_user_repos: bool, emit: Callable) -> list[str]:
    emit(f"Fetching starred repos for @{username}...", event="progress")
    user_stars = fetch_user_stars(username, emit)
    non_forked = [r["full_name"] for r in user_stars if not r.get("fork")]
    forked = [r["full_name"] for r in user_stars if r.get("fork")]

    starred_names = list(non_forked)
    owned_repos, owned_non_forked, owned_forked = [], [], []

    if include_forked or include_user_repos:
        emit("Fetching owned repos...", event="progress")
        owned_repos = fetch_user_owned_repos(username)
        owned_non_forked = [r["full_name"] for r in owned_repos if not r.get("fork")]
        owned_forked = [r["full_name"] for r in owned_repos if r.get("fork")]

    if include_user_repos:
        starred_names.extend(owned_non_forked)
        starred_names.extend(owned_forked)

    added_originals = 0
    if include_forked:
        token = _read_github_token()
        starred_names.extend(forked)
        seen = set(starred_names)
        for fork_name in set(forked + owned_forked):
            original = _fetch_fork_original(fork_name, token)
            if original and original not in seen:
                starred_names.append(original)
                seen.add(original)
                added_originals += 1

    emit(
        f"  Found {len(user_stars)} starred repos ({len(non_forked)} non-forked, {len(forked)} forked)",
        event="progress",
    )
    if include_user_repos:
        emit(f"  Added {len(owned_repos)} owned repos", event="progress")
    if include_forked:
        emit(f"  Added {added_originals} fork originals", event="progress")
    else:
        emit("  Forks excluded (--include-forked not set)", event="progress")

    return list(dict.fromkeys(starred_names))


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
        from math import log
    except ImportError:
        return 1
    min_k = max(2, int(round(log(n))))
    best_k, best_score = 1, -1.0
    for k in range(min_k, upper + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        if int(np.bincount(labels, minlength=k).min()) < 2:
            continue
        try:
            score = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            continue
        if score > best_score:
            best_score, best_k = score, k
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
        from math import log
    except ImportError:
        return 1
    min_k = max(2, int(round(log(n))))
    best_k, best_obj = min_k, None
    for k in range(min_k, upper + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=20, reg_covar=1e-5,
                                  covariance_type="full", init_params="kmeans")
            labels = gmm.fit_predict(embeddings)
            if int(np.bincount(labels, minlength=k).min()) < 2:
                continue
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
            obj = -sil
        except Exception:
            continue
        if best_obj is None or obj < best_obj:
            best_obj, best_k = obj, k
    return best_k if best_obj is not None else 2


def _cluster_starred_embeddings(embeddings: np.ndarray, clustering_type: str, chosen_k: int, force_k: bool = False):
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
        return [0] * n

    labels_nz = None
    if clustering_type == "k-means":
        labels_nz = KMeans(n_clusters=chosen_k, n_init=10, random_state=42).fit_predict(emb_nz)
    elif clustering_type == "agglomerative":
        try:
            agg = AgglomerativeClustering(n_clusters=chosen_k, metric="cosine", linkage="average")
        except TypeError:
            agg = AgglomerativeClustering(n_clusters=chosen_k, affinity="cosine", linkage="average")
        labels_nz = agg.fit_predict(emb_nz)
    elif clustering_type == "gmm":
        k_range = [chosen_k] if force_k else range(chosen_k, 1, -1)
        for k in k_range:
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=20, reg_covar=1e-5,
                                      covariance_type="full", init_params="kmeans")
                cand = gmm.fit_predict(emb_nz)
                if force_k or int(np.bincount(cand, minlength=k).min()) >= 2:
                    labels_nz = cand
                    break
            except Exception:
                continue
        if labels_nz is None:
            labels_nz = np.zeros(len(emb_nz), dtype=np.int32)
    elif clustering_type == "dbscan":
        target_k = max(2, chosen_k)
        best_lbl, best_sc = None, None
        for eps in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
            for ms in sorted(set([2, 3, max(2, n // 20)])):
                lbl = DBSCAN(eps=eps, min_samples=ms, metric="cosine").fit_predict(emb_nz)
                cc = len(set(lbl.tolist()) - {-1})
                if cc <= 0:
                    continue
                sc = (-abs(cc - target_k), int((lbl != -1).sum()))
                if best_sc is None or sc > best_sc:
                    best_sc, best_lbl = sc, lbl
        labels_nz = best_lbl if best_lbl is not None else np.zeros(len(emb_nz), dtype=np.int32)
    else:
        labels_nz = KMeans(n_clusters=chosen_k, n_init=10, random_state=42).fit_predict(emb_nz)

    labels = np.full(n, -1, dtype=np.int32)
    labels[nonzero_idx] = labels_nz
    return labels.tolist()


class RecommendationEngine:
    def __init__(self, progress_cb: Optional[Callable] = None):
        self.progress_cb = progress_cb or (lambda msg, **kw: None)

    def emit(self, message: str, event: str = "progress", **data):
        self.progress_cb(message, event=event, **data)

    def run(
        self,
        username: str,
        top_k: int = 30,
        model_type: str = "nn",
        include_forked: bool = False,
        include_user_repos: bool = False,
        clusters: int = 0,
        clustering_type: str = "k-means",
    ):
        """Run recommendation and emit SSE events. Returns list of result dicts."""
        if not _static.loaded:
            raise RuntimeError("Static data not preloaded. Call preload() first.")

        device = detect_device()
        self.emit(f"Using device: {device}", event="progress")

        starred_names = prepare_starred_repo_names(
            username, include_forked, include_user_repos, self.emit
        )
        self.emit(f"  Using {len(starred_names)} repos for profile building", event="progress")
        for name in starred_names[:10]:
            self.emit(f"    {name}", event="progress")
        if len(starred_names) > 10:
            self.emit(f"    ... and {len(starred_names) - 10} more", event="progress")

        if model_type == "nn":
            return self._run_nn(username, starred_names, top_k, device, include_user_repos, clusters, clustering_type)
        else:
            return self._run_simple(username, starred_names, top_k, include_user_repos)

    def _run_simple(self, username: str, starred_names: list, top_k: int, include_user_repos: bool = False):
        self.emit("\nBuilding user tag profile from starred repos...", event="progress")
        repo_tags_original = dict(_static.repo_tags_original)
        if include_user_repos:
            self.emit("\nEnriching missing repos with LLM-generated tags...", event="progress")
            repo_tags_original = enrich_profile_tags_with_llm(starred_names, repo_tags_original, as_sets=True)
        repo_tags = {n: set(t) if isinstance(t, list) else t for n, t in repo_tags_original.items()}
        user_tag_counts: dict = defaultdict(float)
        known, unknown = [], []

        for name in starred_names:
            if name in repo_tags:
                known.append(name)
                for tag in repo_tags[name]:
                    user_tag_counts[tag] += 1.0
            else:
                unknown.append(name)

        self.emit(f"  {len(known)} repos with tags, {len(unknown)} without", event="progress")
        if not user_tag_counts:
            self.emit("ERROR: No tags found for starred repos!", event="error")
            return []

        total_weight = sum(user_tag_counts.values())
        sorted_tags = sorted(user_tag_counts.items(), key=lambda x: -x[1])
        self.emit(f"  Top tags: {', '.join(f'{t}({int(c)})' for t, c in sorted_tags[:15])}", event="progress")

        self.emit("\nScoring all repos...", event="progress")
        scores, indices = [], []
        starred_set = set(starred_names)
        repo_names = _static.repo_names

        for idx, name in enumerate(repo_names):
            if name in starred_set or name not in repo_tags:
                continue
            score = sum(user_tag_counts[t] / total_weight for t in repo_tags[name] if t in user_tag_counts)
            if score > 0:
                scores.append(score)
                indices.append(idx)

        self.emit(f"  Scored {len(scores)} candidate repos", event="progress")
        if not scores:
            self.emit("ERROR: No candidate repos found!", event="error")
            return []

        scores_arr = np.array(scores, dtype=np.float32)
        indices_arr = np.array(indices, dtype=np.int32)
        actual_k = min(top_k, len(scores))
        topk_mask = np.argpartition(scores_arr, -actual_k)[-actual_k:]
        topk_mask = topk_mask[np.argsort(-scores_arr[topk_mask])]

        results = []
        self.emit(f"\n{'='*60}", event="progress")
        self.emit(f"Top {actual_k} Recommendations for @{username} (Tag-Based)", event="progress")
        self.emit(f"{'='*60}", event="progress")

        for rank, (arr_idx, score) in enumerate(zip(indices_arr[topk_mask], scores_arr[topk_mask]), start=1):
            name = repo_names[arr_idx]
            tags = repo_tags.get(name, set())
            matching = sorted([t for t in tags if t in user_tag_counts], key=lambda t: -user_tag_counts[t])
            r = {
                "rank": rank,
                "name": name,
                "score": float(score),
                "calibrated": None,
                "pct_rank": None,
                "matching_tags": matching[:10],
                "cluster_id": None,
            }
            results.append(r)
            self.emit(
                f"  {rank:2d}. {name}  (score: {score:.4f})",
                event="result",
                **r,
            )
        return results

    def _run_nn(self, username: str, starred_names: list, top_k: int, device: str, include_user_repos: bool, clusters: int, clustering_type: str):
        repo_names = _static.repo_names
        # work on a local copy so LLM enrichment doesn't mutate shared state
        repo_tags_original = dict(_static.repo_tags_original)
        tag2idx = _static.tag2idx
        model = _static.model
        tag_indices = _static.tag_indices
        tag_weights = _static.tag_weights

        self.emit(f"\nMoving model to {device}...", event="progress")
        try:
            model = model.to(device)
            ti = tag_indices.to(device)
            tw = tag_weights.to(device)
        except Exception as e:
            self.emit(f"  WARNING: could not use {device}, falling back to cpu: {e}", event="progress")
            device = "cpu"
            model = model.to(device)
            ti = tag_indices.to(device)
            tw = tag_weights.to(device)

        self.emit("Computing all repo embeddings...", event="progress")
        with torch.no_grad():
            repo_embs = model(ti, tw)

        if include_user_repos:
            self.emit("\nEnriching missing repos with LLM-generated tags...", event="progress")
            repo_tags_original = enrich_profile_tags_with_llm(
                starred_names, repo_tags_original, as_sets=False
            )

        repo_tags_sets = {n: set(t) for n, t in repo_tags_original.items()}

        self.emit("\nBuilding user embedding from starred repos...", event="progress")
        starred_set = set(starred_names)
        name_to_idx = {name: idx for idx, name in enumerate(repo_names)}
        known_idx, unknown = [], []
        for name in starred_names:
            if name in name_to_idx:
                known_idx.append(name_to_idx[name])
            else:
                unknown.append(name)

        self.emit(f"  {len(known_idx)} found in graph, {len(unknown)} not found", event="progress")

        extra_embs = None
        unknown_with_tags = [n for n in unknown if n in repo_tags_original and repo_tags_original[n]]
        if unknown_with_tags:
            extra_ti, extra_tw = build_repo_tag_tensors(unknown_with_tags, repo_tags_original, tag2idx)
            extra_ti, extra_tw = extra_ti.to(device), extra_tw.to(device)
            with torch.no_grad():
                extra_embs = model(extra_ti, extra_tw)
            self.emit(f"  {len(unknown_with_tags)} unknown repos computed on-the-fly", event="progress")

        if not known_idx and extra_embs is None:
            if unknown:
                self.emit(
                    f"ERROR: None of the {len(unknown)} starred/owned repos have tags. "
                    "Try enabling --include-user-repos to generate LLM tags for them.",
                    event="error",
                )
            else:
                self.emit("ERROR: No starred repos found in graph!", event="error")
            return []

        parts = []
        starred_ordered = []
        if known_idx:
            parts.append(repo_embs[torch.tensor(known_idx, device=device)])
            starred_ordered.extend([repo_names[i] for i in known_idx])
        if extra_embs is not None:
            parts.append(extra_embs)
            starred_ordered.extend(unknown_with_tags)

        starred_embs = torch.cat(parts, dim=0)
        keep_mask = torch.linalg.vector_norm(starred_embs, dim=1) > 1e-12
        zero_count = int((~keep_mask).sum().item())
        if zero_count:
            self.emit(f"  WARNING: {zero_count} starred repos have zero embeddings, excluded", event="progress")
        if int(keep_mask.sum().item()) == 0:
            self.emit("ERROR: All starred repo embeddings are zero.", event="error")
            return []
        if zero_count:
            keep_list = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()
            starred_embs = starred_embs[keep_mask]
            starred_ordered = [starred_ordered[i] for i in keep_list]

        n_starred = int(starred_embs.shape[0])

        user_tag_counts: dict = defaultdict(float)
        for name in set(starred_ordered):
            for tag in repo_tags_sets.get(name, set()):
                user_tag_counts[tag] += 1.0

        if clusters < 0:
            clusters = 0
        user_requested_k = False
        if clusters == 1 or n_starred < 6:
            chosen_k = 1
        elif clusters > 1:
            chosen_k = min(clusters, n_starred)
            user_requested_k = True
        else:
            emb_np = starred_embs.detach().cpu().numpy()
            if clustering_type == "gmm":
                chosen_k = _choose_k_gmm(emb_np, max_k=8)
            else:
                chosen_k = _choose_k(emb_np, max_k=8)

        self.emit(f"  Clustering type: {clustering_type}", event="progress")
        label = "User-requested" if user_requested_k else "Auto-selected"
        self.emit(f"  {label} {chosen_k} cluster(s)", event="progress")

        norm_repo_embs = F.normalize(repo_embs, dim=-1)
        excluded_idx = set(known_idx)
        valid_mask_base = torch.ones(norm_repo_embs.shape[0], dtype=torch.bool, device=device)
        if excluded_idx:
            valid_mask_base[torch.tensor(sorted(excluded_idx), device=device)] = False

        actual_top_k = min(top_k, int(valid_mask_base.sum().item()))

        if chosen_k == 1:
            cluster_ids = [0] * n_starred
        else:
            cluster_ids = _cluster_starred_embeddings(
                starred_embs.detach().cpu().numpy(), clustering_type, chosen_k,
                force_k=user_requested_k
            )

        cluster_members: dict = defaultdict(list)
        cluster_member_idx: dict = defaultdict(list)
        for i, cid in enumerate(cluster_ids):
            if int(cid) < 0:
                continue
            cluster_members[int(cid)].append(starred_ordered[i])
            cluster_member_idx[int(cid)].append(i)

        cluster_order = sorted(cluster_members.keys(), key=lambda c: -len(cluster_members[c]))
        if not cluster_order:
            cluster_order = [0]
            cluster_members = defaultdict(list, {0: list(starred_ordered)})
            cluster_member_idx = defaultdict(list, {0: list(range(n_starred))})

        centroids = []
        for cid in cluster_order:
            midx = cluster_member_idx[cid]
            centroids.append(F.normalize(starred_embs[midx].mean(dim=0) if midx else starred_embs.mean(dim=0), dim=-1))
        cluster_count = len(cluster_order)
        centroids_tensor = torch.stack(centroids, dim=0)
        cluster_scores_matrix = centroids_tensor @ norm_repo_embs.t()
        if excluded_idx:
            cluster_scores_matrix[:, torch.tensor(sorted(excluded_idx), device=device)] = -1e9

        base = actual_top_k // max(1, cluster_count)
        rem = actual_top_k % max(1, cluster_count)
        cluster_quota = {cid: base + (1 if rank < rem else 0) for rank, cid in enumerate(cluster_order)}

        self.emit(f"\n{'='*60}", event="progress")
        self.emit(f"Top {actual_top_k} Recommendations for @{username} (NN Clustered)", event="progress")
        self.emit(f"{'='*60}", event="progress")

        results = []
        recommended_global: set = set()

        for c_rank, cid in enumerate(cluster_order, start=1):
            member_names = cluster_members[cid]
            tag_counts: dict = defaultdict(float)
            for name in member_names:
                for tag in repo_tags_sets.get(name, set()):
                    tag_counts[tag] += 1.0
            top_cluster_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])[:5]]
            tags_str = ", ".join(top_cluster_tags) or "(no tags)"

            cluster_info = {
                "cluster_id": c_rank,
                "cluster_total": cluster_count,
                "member_count": len(member_names),
                "top_tags": top_cluster_tags,
                "member_names": member_names,
            }
            self.emit(
                f"\n[Cluster {c_rank}/{cluster_count}] members={len(member_names)} top-tags: {tags_str}",
                event="cluster",
                **cluster_info,
            )

            quota = cluster_quota.get(cid, 0)
            if quota <= 0:
                continue
            scores = cluster_scores_matrix[c_rank - 1].clone()
            if recommended_global:
                scores[torch.tensor(sorted(recommended_global), device=device)] = -1e9

            valid_scores = scores[scores > -1e8]
            if valid_scores.numel() == 0:
                self.emit("  No candidates left after deduplication.", event="progress")
                continue
            take = min(quota, int(valid_scores.numel()))
            topk_scores, topk_indices = scores.topk(take)
            sorted_valid = torch.sort(valid_scores).values
            valid_mean = valid_scores.mean()
            valid_std = valid_scores.std(unbiased=False).clamp(min=1e-8)

            for rank, (idx, score) in enumerate(zip(topk_indices.tolist(), topk_scores.tolist()), start=1):
                recommended_global.add(idx)
                name = repo_names[idx]
                tags = repo_tags_sets.get(name, set())
                matching = sorted([t for t in tags if t in tag_counts], key=lambda t: -tag_counts[t])
                score_t = torch.tensor(score, device=device)
                z = (score_t - valid_mean) / valid_std
                calibrated = float(torch.sigmoid(z).item())
                pct_rank = float(
                    torch.searchsorted(sorted_valid, score_t, right=True).float() / float(sorted_valid.numel())
                )
                r = {
                    "rank": len(results) + 1,
                    "name": name,
                    "score": float(score),
                    "calibrated": calibrated,
                    "pct_rank": pct_rank,
                    "matching_tags": matching[:10],
                    "cluster_id": c_rank,
                    "source": "cluster",
                }
                results.append(r)
                self.emit(
                    f"  {rank:2d}. {name}  (score: {score:.4f}, cal: {calibrated:.4f}, pct: {pct_rank:.4f})",
                    event="result",
                    **r,
                )
                if matching:
                    self.emit(f"      Tags: {', '.join(matching[:10])}", event="progress")

        self.emit("\nComputing overall recommendations (global profile)...", event="progress")

        user_emb = F.normalize(starred_embs.sum(dim=0, keepdim=True), dim=-1)
        overall_scores = (user_emb @ norm_repo_embs.t()).squeeze(0)
        if excluded_idx:
            overall_scores[torch.tensor(sorted(excluded_idx), device=device)] = -1e9
        valid_ov = overall_scores[overall_scores > -1e8]
        if valid_ov.numel() == 0:
            return results
        ov_k = min(actual_top_k, int(valid_ov.numel()))
        ov_scores, ov_indices = overall_scores.topk(ov_k)
        sorted_ov = torch.sort(valid_ov).values
        ov_mean = valid_ov.mean()
        ov_std = valid_ov.std(unbiased=False).clamp(min=1e-8)
        ov_quantiles = torch.quantile(valid_ov, torch.tensor([0.5, 0.9, 0.99], device=device))
        self.emit(
            f"Score spread p50/p90/p99: {ov_quantiles[0]:.4f}/{ov_quantiles[1]:.4f}/{ov_quantiles[2]:.4f}",
            event="progress",
        )

        for rank, (idx, score) in enumerate(zip(ov_indices.tolist(), ov_scores.tolist()), start=1):
            name = repo_names[idx]
            tags = repo_tags_sets.get(name, set())
            matching = sorted([t for t in tags if t in user_tag_counts], key=lambda t: -user_tag_counts[t])
            score_t = torch.tensor(score, device=device)
            z = (score_t - ov_mean) / ov_std
            calibrated = float(torch.sigmoid(z).item())
            pct_rank = float(
                torch.searchsorted(sorted_ov, score_t, right=True).float() / float(sorted_ov.numel())
            )
            r = {
                "rank": rank,
                "name": name,
                "score": float(score),
                "calibrated": calibrated,
                "pct_rank": pct_rank,
                "matching_tags": matching[:10],
                "cluster_id": None,
                "source": "overall",
            }
            self.emit(
                f"  {rank:2d}. {name}  (score: {score:.4f}, cal: {calibrated:.4f}, pct: {pct_rank:.4f})",
                event="result",
                **r,
            )

        return results
