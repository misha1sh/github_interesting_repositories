import asyncio
import json
import os
import re
import time

import requests


TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github_token")
CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_token() -> str:
    with open(TOKEN_PATH) as f:
        return f.read().strip()


def load_claude_env() -> dict[str, str]:
    if not os.path.exists(CLAUDE_SETTINGS_PATH):
        return {}
    with open(CLAUDE_SETTINGS_PATH) as f:
        settings = json.load(f)
    return settings.get("env", {})


def gh_get_json(url: str, token: str):
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
    )
    return r.json()


def fetch_repo_context(repo: str, token: str) -> str:
    log(f"  pre-fetching metadata+readme+tree for {repo}")

    meta = gh_get_json(f"https://api.github.com/repos/{repo}", token)
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
        contents = gh_get_json(f"https://api.github.com/repos/{repo}/contents", token)
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
            sub = gh_get_json(f"https://api.github.com/repos/{repo}/contents/{subdir}", token)
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


def fetch_file_content(repo: str, path: str) -> str:
    try:
        r = requests.get(
            f"https://raw.githubusercontent.com/{repo}/HEAD/{path}", timeout=10,
        )
        if r.status_code != 200:
            return f"(HTTP {r.status_code})"
        return "\n".join(r.content.decode("utf-8", errors="replace").splitlines()[:100])
    except Exception as e:
        return f"(error: {e})"


def extract_text(message) -> list[str]:
    out = []
    for attr in ("content", "text", "result"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, list):
            for block in val:
                t = getattr(block, "text", None)
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
    return out


def parse_json_array(text: str) -> list[str]:
    m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    return [
        re.sub(r"[^a-z0-9+._-]+", "-", str(t).strip().lower()).strip("-")
        for t in arr if isinstance(t, str) and t.strip()
    ][:7]


def parse_file_list(text: str) -> list[str]:
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


_sdk_options = None


def get_sdk_options():
    global _sdk_options
    if _sdk_options is not None:
        return _sdk_options
    from claude_agent_sdk import ClaudeAgentOptions
    claude_env = load_claude_env()
    _sdk_options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=1,
        model="haiku",
        env=claude_env,
    )
    log("SDK options created (singleton, model=haiku, no tools)")
    return _sdk_options


async def sdk_query(prompt: str) -> str:
    from claude_agent_sdk import query
    options = get_sdk_options()
    chunks = []
    async for msg in query(prompt=prompt, options=options):
        chunks.extend(extract_text(msg))
    return "\n".join(chunks).strip()


async def generate_tags_for_repo(repo: str) -> tuple[list[str], str]:
    token = read_token()
    context = fetch_repo_context(repo, token)

    log(f"  turn 1: asking LLM which files to read...")
    t0 = time.time()
    file_request = await sdk_query(
        f"Here is a GitHub repository overview:\n\n{context}\n\n"
        "Pick up to 3 source files from the root directory listing that best reveal what this project does.\n"
        "Skip README, LICENSE, configs, .gitignore, lockfiles.\n"
        "If the root only has directories, pick paths like dir/likely_main_file.ext.\n"
        "Respond with ONLY a JSON array of file paths, nothing else.\n"
        'Example: ["main.cpp", "src/app.py", "lib/utils.ts"]'
    )
    log(f"  turn 1 done in {time.time()-t0:.1f}s: {file_request[:200]}")

    file_paths = parse_file_list(file_request)
    log(f"  files to fetch: {file_paths}")

    file_sections = ""
    for path in file_paths:
        log(f"  fetching {path}...")
        content = fetch_file_content(repo, path)
        file_sections += f"\n=== {path} (first 100 lines) ===\n{content}\n"

    log(f"  turn 2: asking LLM for tags...")
    t1 = time.time()
    tag_response = await sdk_query(
        f"Here is a GitHub repository with its source code:\n\n{context}\n"
        f"{file_sections}\n\n"
        "Based on all of the above, return ONLY a JSON array of 3-7 lowercase tags categorizing this repository.\n"
        'Example: ["python", "web", "api"]'
    )
    log(f"  turn 2 done in {time.time()-t1:.1f}s: {tag_response[:200]}")

    tags = parse_json_array(tag_response)
    log(f"  => {repo}: {tags}")
    return tags, tag_response


async def run(repos: list[str]):
    for repo in repos:
        t0 = time.time()
        try:
            tags, raw = await generate_tags_for_repo(repo)
            log(f"  DONE in {time.time()-t0:.1f}s: {tags}")
        except BaseException as e:
            log(f"  FAILED in {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        log("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repos", nargs="+")
    args = parser.parse_args()
    asyncio.run(run(args.repos))
