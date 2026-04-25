import argparse
import json
import os
import subprocess

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from model import InductivePinSage
from train import load_data, precompute_graph_tensors

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github_token")


def fetch_user_stars(username: str) -> list[dict]:
    with open(TOKEN_PATH) as f:
        token = f.read().strip()

    all_repos = []
    page = 1
    while True:
        result = subprocess.run(
            ["curl", "-s",
             "-H", f"Authorization: Bearer {token}",
             "-H", "Accept: application/vnd.github+json",
             f"https://api.github.com/users/{username}/starred?per_page=100&page={page}"],
            capture_output=True, text=True
        )
        repos = json.loads(result.stdout)
        if not repos or not isinstance(repos, list):
            break
        all_repos.extend(repos)
        if len(repos) < 100:
            break
        page += 1

    return [{"full_name": r["full_name"], "stargazers_count": r["stargazers_count"]} for r in all_repos]


def embed_repo_names(names: list[str], model_name="nomic-ai/nomic-embed-text-v1.5") -> torch.Tensor:
    model = SentenceTransformer(model_name, trust_remote_code=True)
    prefixed = [f"search_document: {n}" for n in names]
    embs = model.encode(prefixed, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return torch.from_numpy(embs).float()


def recommend(username: str, top_k: int = 30, device: str = "cuda"):
    print(f"Fetching starred repos for {username}...")
    user_stars = fetch_user_stars(username)
    starred_names = [r["full_name"] for r in user_stars]
    print(f"  Found {len(starred_names)} starred repos")
    for name in starred_names[:10]:
        print(f"    {name}")
    if len(starred_names) > 10:
        print(f"    ... and {len(starred_names) - 10} more")

    print("\nLoading m  odel and graph data...")
    data, meta = load_data()
    repo_names = meta["repo_names"]
    repo2idx = meta["repo2idx"]
    unique_repos = meta["unique_repos"]

    gd = precompute_graph_tensors(data, device)

    model = InductivePinSage(
        nomic_dim=768, hidden_dim=128, num_hops=4,
        dropout=0.0, conv_type="gat", gat_heads=4,
        use_layernorm=True, normalize_output=False,
    ).to(device)

    ckpt_path = os.path.join(DATA_DIR, "best_model.pt")
    with torch.no_grad():
        model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])
    state_dict = torch.load(ckpt_path, weights_only=True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    print("Computing graph embeddings...")
    with torch.no_grad():
        h_user, h_repo = model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])

    known_repo_indices = []
    unknown_starred = []
    for name in starred_names:
        found = False
        for rid, idx in repo2idx.items():
            if repo_names[idx] == name:
                known_repo_indices.append(idx)
                found = True
                break
        if not found:
            unknown_starred.append(name)

    print(f"  {len(known_repo_indices)} starred repos found in graph, {len(unknown_starred)} unknown")

    if unknown_starred:
        print("  Embedding unknown repos with Nomic...")
        unknown_embs = embed_repo_names(unknown_starred).to(device)
        unknown_stars = torch.ones(len(unknown_starred), 1, device=device)
        with torch.no_grad():
            unknown_repo_features = model.repo_encoder(unknown_embs, unknown_stars)
    else:
        unknown_repo_features = None

    with torch.no_grad():
        x_repo_encoded = model.repo_encoder(gd["x_nomic"], gd["x_stars"])

    all_repo_features = []
    if known_repo_indices:
        all_repo_features.append(x_repo_encoded[torch.tensor(known_repo_indices, device=device)])
    if unknown_repo_features is not None:
        all_repo_features.append(unknown_repo_features)

    if not all_repo_features:
        print("ERROR: No starred repos could be processed!")
        return

    starred_features = torch.cat(all_repo_features, dim=0)
    user_emb = starred_features.mean(dim=0, keepdim=True)

    user_emb_norm = F.normalize(user_emb, dim=-1)
    repo_embs_norm = F.normalize(h_repo, dim=-1)
    scores = (user_emb_norm @ repo_embs_norm.t()).squeeze(0)

    exclude_set = set(known_repo_indices)
    for idx in exclude_set:
        scores[idx] = -1e9

    topk_scores, topk_indices = scores.topk(top_k)

    print(f"\n{'='*70}")
    print(f"Top {top_k} Recommended Repositories for @{username}")
    print(f"{'='*70}")
    for rank, (idx, score) in enumerate(zip(topk_indices.tolist(), topk_scores.tolist())):
        name = repo_names[idx]
        print(f"  {rank+1:2d}. {name:<50s}  (score: {score:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    recommend(args.username, args.top_k, args.device)


if __name__ == "__main__":
    main()
