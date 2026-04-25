import os

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "star_events_2025.csv")
OUT_DIR = os.path.join(DATA_DIR, "processed")


def load_csv(path: str):
    print(f"  Reading {path} with pandas...")
    df = pd.read_csv(path, dtype={"user_id": "int32", "repo_id": "int32", "repo_name": str})
    df.drop_duplicates(subset=["user_id", "repo_id"], inplace=True)
    print(f"  {len(df)} rows after dedup")
    return df["user_id"].tolist(), df["repo_id"].tolist(), df["repo_name"].tolist()


def build_mappings(user_ids, repo_ids, repo_names):
    unique_users = sorted(set(user_ids))
    unique_repos = sorted(set(repo_ids))
    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    repo2idx = {rid: i for i, rid in enumerate(unique_repos)}

    repo_id_to_name = {}
    for rid, rname in zip(repo_ids, repo_names):
        repo_id_to_name[rid] = rname

    ordered_repo_names = [repo_id_to_name[rid] for rid in unique_repos]
    return user2idx, repo2idx, unique_users, unique_repos, ordered_repo_names


def compute_nomic_embeddings(repo_names: list[str], batch_size: int = 512) -> torch.Tensor:
    print(f"Computing Nomic embeddings for {len(repo_names)} repos...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    prefixed = [f"search_document: {name}" for name in repo_names]
    embeddings = model.encode(
        prefixed,
        batch_size=2048,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return torch.from_numpy(embeddings).float()


def build_graph(user_ids, repo_ids, user2idx, repo2idx, nomic_embs: torch.Tensor):
    src = torch.tensor([user2idx[u] for u in user_ids], dtype=torch.long)
    dst = torch.tensor([repo2idx[r] for r in repo_ids], dtype=torch.long)

    num_users = len(user2idx)
    num_repos = len(repo2idx)

    repo_degree = torch.zeros(num_repos, dtype=torch.float)
    for r in repo_ids:
        repo_degree[repo2idx[r]] += 1.0
    log_star_count = torch.log1p(repo_degree).unsqueeze(1)

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["repo"].num_nodes = num_repos
    data["repo"].x_nomic = nomic_embs
    data["repo"].x_stars = log_star_count
    data["user", "stars", "repo"].edge_index = torch.stack([src, dst], dim=0)

    data = T.ToUndirected()(data)

    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CSV...")
    user_ids, repo_ids, repo_names = load_csv(CSV_PATH)
    print(f"  {len(user_ids)} events, {len(set(user_ids))} users, {len(set(repo_ids))} repos")

    print("Building mappings...")
    user2idx, repo2idx, unique_users, unique_repos, ordered_repo_names = build_mappings(
        user_ids, repo_ids, repo_names
    )

    emb_path = os.path.join(OUT_DIR, "repo_nomic_embs.pt")
    if os.path.exists(emb_path):
        print(f"Loading cached Nomic embeddings from {emb_path}")
        nomic_embs = torch.load(emb_path, weights_only=True)
    else:
        nomic_embs = compute_nomic_embeddings(ordered_repo_names)
        torch.save(nomic_embs, emb_path)
        print(f"Saved Nomic embeddings to {emb_path}")

    print("Building HeteroData graph...")
    data = build_graph(user_ids, repo_ids, user2idx, repo2idx, nomic_embs)
    print(f"  {data}")

    graph_path = os.path.join(OUT_DIR, "graph.pt")
    torch.save(data, graph_path)
    print(f"Saved graph to {graph_path}")

    meta = {
        "user2idx": user2idx,
        "repo2idx": repo2idx,
        "unique_users": unique_users,
        "unique_repos": unique_repos,
        "repo_names": ordered_repo_names,
    }
    meta_path = os.path.join(OUT_DIR, "meta.pt")
    torch.save(meta, meta_path)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
