#!/usr/bin/env python3
"""
Collect GitHub repo stats + READMEs using GraphQL batch queries.

Optimizations:
- GraphQL aliases: 50 repos per request (50x fewer calls than REST)
- 5 README filename variants attempted in same query
- Async concurrency (4 parallel) to use available rate-limit headroom
- Secondary rate-limit detection: HTTP 200 with error body is handled
- Recovery: skips repos already in repo_stats.csv or readmes/

Recovery: on restart, skips repos already in repo_stats.csv or readmes/.
"""
import asyncio
import csv
import json
import os
import time
import base64
import aiohttp
from datetime import datetime, timedelta

TOKEN_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.github_token")
CSV_INPUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_repos_100plus_stars.csv")
CSV_OUTPUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo_stats.csv")
READMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readmes")

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL    = "https://api.github.com"

BATCH_SIZE        = 50   # repos per GraphQL query
CONCURRENCY       = 4    # simultaneous in-flight requests
MIN_RL_LEFT       = 100  # pause if primary rate-limit remaining drops below this
SECONDARY_RL_WAIT = 90   # seconds to wait when secondary rate limit is hit

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def build_graphql_query(repos):
    parts = []
    for i, (repo_id, repo_name) in enumerate(repos):
        owner, name = repo_name.split("/", 1)
        owner = owner.replace('"', '\\"')
        name  = name.replace('"', '\\"')
        parts.append(f"""  r{i}: repository(owner: "{owner}", name: "{name}") {{
    stargazerCount forkCount
    watchers {{ totalCount }}
    defaultBranchRef {{
      target {{ ... on Commit {{ history {{ totalCount }} }} }}
    }}
    readme_md:  object(expression: "HEAD:README.md")  {{ ... on Blob {{ text isTruncated }} }}
    readme_MD:  object(expression: "HEAD:README.MD")  {{ ... on Blob {{ text isTruncated }} }}
    readme_rst: object(expression: "HEAD:README.rst") {{ ... on Blob {{ text isTruncated }} }}
    readme_txt: object(expression: "HEAD:README.txt") {{ ... on Blob {{ text isTruncated }} }}
    readme_low: object(expression: "HEAD:readme.md")  {{ ... on Blob {{ text isTruncated }} }}
  }}""")
    return "{\n" + "\n".join(parts) + "\n}"


def extract_readme(repo_data):
    for key in ("readme_md", "readme_MD", "readme_rst", "readme_txt", "readme_low"):
        obj = repo_data.get(key)
        if obj and obj.get("text"):
            return obj["text"]
    return None


def load_processed_ids():
    processed = set()
    if os.path.exists(CSV_OUTPUT):
        with open(CSV_OUTPUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    processed.add(int(row["repo_id"]))
                except (KeyError, ValueError):
                    pass
    if os.path.isdir(READMES_DIR):
        for fname in os.listdir(READMES_DIR):
            try:
                processed.add(int(fname))
            except ValueError:
                pass
    return processed


class RateLimiter:
    def __init__(self):
        self.remaining = 5000
        self.reset_at  = 0
        self._lock     = asyncio.Lock()
        self.secondary_hits = 0

    def update(self, headers):
        try:
            self.remaining = int(headers.get("X-RateLimit-Remaining", self.remaining))
            self.reset_at  = int(headers.get("X-RateLimit-Reset",     self.reset_at))
        except (TypeError, ValueError):
            pass

    async def wait_primary(self):
        async with self._lock:
            if self.remaining < MIN_RL_LEFT:
                wait = max(self.reset_at - time.time(), 0) + 10
                reset_str = datetime.fromtimestamp(self.reset_at).strftime("%H:%M:%S")
                print(f"\n  Primary rate limit low ({self.remaining}). Waiting {wait:.0f}s until {reset_str}...",
                      flush=True)
                await asyncio.sleep(wait)

    async def wait_secondary(self):
        self.secondary_hits += 1
        wait = SECONDARY_RL_WAIT
        print(f"\n  Secondary rate limit hit (#{self.secondary_hits}). Waiting {wait}s...", flush=True)
        await asyncio.sleep(wait)


class Counter:
    def __init__(self, total):
        self.total      = total
        self.done       = 0
        self.start_time = time.time()
        self._lock      = asyncio.Lock()

    async def add(self, n):
        async with self._lock:
            self.done += n

    def print_progress(self, rl: RateLimiter):
        elapsed = time.time() - self.start_time
        rate    = self.done / elapsed if elapsed > 0 else 0
        eta_s   = (self.total - self.done) / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_s)))
        pct     = self.done / self.total * 100
        print(
            f"\r[{pct:5.1f}%] {self.done:>7}/{self.total}  "
            f"{rate:5.1f} repos/s  ETA {eta_str}  RL {rl.remaining}   ",
            end="", flush=True,
        )


def is_secondary_rate_limit(body) -> bool:
    if not isinstance(body, dict):
        return False
    msg = body.get("message", "")
    return "secondary rate limit" in msg.lower() if isinstance(msg, str) else False


async def process_batch(
    session: aiohttp.ClientSession,
    batch: list,
    rl: RateLimiter,
    counter: Counter,
    rest_queue: list,
    csv_writer,
    csv_lock: asyncio.Lock,
):
    await rl.wait_primary()

    query = build_graphql_query(batch)
    data  = {}

    for attempt in range(6):
        try:
            async with session.post(
                GRAPHQL_URL,
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                rl.update(resp.headers)
                try:
                    body = await resp.json(content_type=None)
                except (json.JSONDecodeError, ValueError):
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue

                if resp.status == 200 and not is_secondary_rate_limit(body):
                    data = body.get("data") or {}
                    break

                if is_secondary_rate_limit(body) or resp.status in (403, 429):
                    await rl.wait_secondary()
                    continue

                await asyncio.sleep(min(2 ** attempt, 30))

        except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
            await asyncio.sleep(10)
        except aiohttp.ClientError as e:
            print(f"\n  aiohttp error: {e}")
            await asyncio.sleep(10)

    rows = []
    readmes_to_write = []

    for i, (repo_id, repo_name) in enumerate(batch):
        repo_data = data.get(f"r{i}")

        if repo_data is None:
            # Repo not found / private / truly errored — skip writing so it's retried
            continue

        stars    = repo_data.get("stargazerCount")
        forks    = repo_data.get("forkCount")
        watchers = (repo_data.get("watchers") or {}).get("totalCount")

        commits    = None
        branch_ref = repo_data.get("defaultBranchRef")
        if branch_ref and branch_ref.get("target"):
            commits = (branch_ref["target"].get("history") or {}).get("totalCount")

        readme_text = extract_readme(repo_data)

        if readme_text:
            readmes_to_write.append((os.path.join(READMES_DIR, str(repo_id)), readme_text))
        else:
            rest_queue.append((repo_id, repo_name))

        rows.append({
            "repo_id": repo_id, "repo_name": repo_name,
            "stars": stars, "forks": forks, "watchers": watchers,
            "commits": commits, "has_readme": readme_text is not None,
        })

    for path, text in readmes_to_write:
        with open(path, "w", encoding="utf-8") as rf:
            rf.write(text)

    if rows:
        async with csv_lock:
            csv_writer.writerows(rows)

    await counter.add(len(batch))
    counter.print_progress(rl)


async def process_rest_readme(
    session: aiohttp.ClientSession,
    repo_id: int,
    repo_name: str,
    rl: RateLimiter,
    done_counter: list,
    total_rest: int,
    start_rest: float,
):
    await rl.wait_primary()
    owner, name = repo_name.split("/", 1)
    url = f"{REST_URL}/repos/{owner}/{name}/readme"

    for attempt in range(6):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                rl.update(resp.headers)
                try:
                    body = await resp.json(content_type=None)
                except (json.JSONDecodeError, ValueError):
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue

                if isinstance(body, dict) and is_secondary_rate_limit(body):
                    await rl.wait_secondary()
                    continue

                if resp.status == 200:
                    content  = body.get("content", "")
                    encoding = body.get("encoding", "")
                    if encoding == "base64":
                        try:
                            text = base64.b64decode(content).decode("utf-8", errors="replace")
                        except Exception:
                            text = None
                    else:
                        text = content or None
                    if text:
                        with open(os.path.join(READMES_DIR, str(repo_id)), "w", encoding="utf-8") as rf:
                            rf.write(text)
                    break

                if resp.status == 404:
                    break

                if resp.status in (403, 429):
                    await rl.wait_secondary()
                    continue

                await asyncio.sleep(min(2 ** attempt, 30))

        except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
            await asyncio.sleep(10)
        except aiohttp.ClientError as e:
            await asyncio.sleep(10)

    done_counter[0] += 1
    n    = done_counter[0]
    rate = n / max(time.time() - start_rest, 0.001)
    eta  = str(timedelta(seconds=int((total_rest - n) / rate))) if rate > 0 else "?"
    print(f"\r  REST [{n}/{total_rest}]  {rate:.1f}/s  ETA {eta}  RL {rl.remaining}   ",
          end="", flush=True)


async def main_async():
    os.makedirs(READMES_DIR, exist_ok=True)

    print("Loading processed repo IDs...")
    processed_ids = load_processed_ids()
    print(f"  Already processed: {len(processed_ids)} repos")

    all_repos = []
    with open(CSV_INPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                repo_id = int(row["repo_id"])
            except (KeyError, ValueError):
                continue
            if repo_id not in processed_ids:
                all_repos.append((repo_id, row["repo_name"]))

    total = len(all_repos)
    print(f"  Repos to process: {total}")
    if not total:
        print("All repos already processed!")
        return

    csv_exists = os.path.exists(CSV_OUTPUT)
    csv_file   = open(CSV_OUTPUT, "a", newline="", encoding="utf-8")
    fieldnames = ["repo_id", "repo_name", "stars", "forks", "watchers", "commits", "has_readme"]
    writer     = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    rl         = RateLimiter()
    counter    = Counter(total)
    csv_lock   = asyncio.Lock()
    semaphore  = asyncio.Semaphore(CONCURRENCY)
    rest_queue = []

    async def bounded_batch(batch):
        async with semaphore:
            await process_batch(session, batch, rl, counter, rest_queue, writer, csv_lock)

    print(f"\nStarting GraphQL phase  (batch={BATCH_SIZE}, concurrency={CONCURRENCY})...\n")

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        batches = [all_repos[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
        await asyncio.gather(*[asyncio.create_task(bounded_batch(b)) for b in batches])

        csv_file.flush()
        print(f"\n\nGraphQL phase complete. REST fallback queue: {len(rest_queue)} repos.\n")

        if rest_queue:
            print(f"Fetching {len(rest_queue)} READMEs via REST API...")
            done_counter = [0]
            start_rest   = time.time()
            rest_sem     = asyncio.Semaphore(CONCURRENCY)

            async def bounded_rest(repo_id, repo_name):
                async with rest_sem:
                    await process_rest_readme(session, repo_id, repo_name, rl,
                                              done_counter, len(rest_queue), start_rest)

            await asyncio.gather(*[
                asyncio.create_task(bounded_rest(rid, rname))
                for rid, rname in rest_queue
            ])
            print(f"\n  REST phase complete.")

    csv_file.close()
    elapsed = time.time() - counter.start_time
    print(f"\nAll done!  {counter.done} repos processed in {str(timedelta(seconds=int(elapsed)))}")
    print(f"CSV     : {CSV_OUTPUT}")
    print(f"READMEs : {READMES_DIR}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
