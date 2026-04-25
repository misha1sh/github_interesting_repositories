import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "star_events_2025.csv")
TAGS_DIR = os.path.join(os.path.dirname(DATA_DIR), "1_3_repo_tags")
REPOS_CSV = os.path.join(os.path.dirname(DATA_DIR), "1_2_collect_api", "github_repos_100plus_stars.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_data():
    data = torch.load(os.path.join(PROCESSED_DIR, "graph.pt"), weights_only=False)
    meta = torch.load(os.path.join(PROCESSED_DIR, "meta.pt"), weights_only=False)
    return data, meta


def load_tags():
    print("  Loading repo name mapping...")
    repos_df = pd.read_csv(REPOS_CSV)
    row_id_to_name = {row['repo_id']: row['repo_name'] for _, row in repos_df.iterrows()}
    
    print("  Loading tags...")
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
                    repo_tags[repo_name] = set(data["tags"])
    return repo_tags


def load_csv(path):
    df = pd.read_csv(path, dtype={"user_id": "int32", "repo_id": "int32"})
    df.drop_duplicates(subset=["user_id", "repo_id"], inplace=True)
    return df["user_id"].tolist(), df["repo_id"].tolist()


def build_mappings(user_ids, repo_ids):
    unique_users = sorted(set(user_ids))
    unique_repos = sorted(set(repo_ids))
    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    repo2idx = {rid: i for i, rid in enumerate(unique_repos)}
    return user2idx, repo2idx, unique_users, unique_repos


def train_val_test_split(data, seed=42):
    edge_index = data["user", "stars", "repo"].edge_index
    E = edge_index.size(1)
    users_np = edge_index[0].numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    order = np.argsort(users_np, kind="stable")
    sorted_users = users_np[order]

    splits = np.flatnonzero(np.diff(sorted_users)) + 1
    group_starts = np.concatenate([[0], splits])
    group_ends = np.concatenate([splits, [E]])
    counts = group_ends - group_starts

    test_offsets = (rng.random(len(counts)) * counts).astype(np.int64)
    test_global = group_starts + test_offsets
    test_edge_indices = order[test_global]

    has_val = counts >= 3
    val_starts = group_starts[has_val]
    val_counts = counts[has_val]
    val_test_offsets = test_offsets[has_val]

    val_raw = (rng.random(len(val_counts)) * (val_counts - 1)).astype(np.int64)
    val_adj = np.where(val_raw >= val_test_offsets, val_raw + 1, val_raw)
    val_global = val_starts + val_adj
    val_edge_indices = order[val_global]

    train_mask = np.ones(E, dtype=bool)
    train_mask[test_edge_indices] = False
    train_mask[val_edge_indices] = False

    return (
        torch.from_numpy(np.flatnonzero(train_mask)),
        torch.from_numpy(val_edge_indices),
        torch.from_numpy(test_edge_indices),
    )


def build_csr(edge_index, num_nodes, dim=0):
    ids = edge_index[dim].numpy().astype(np.int64)
    other = edge_index[1 - dim].numpy().astype(np.int64)
    order = np.argsort(ids, kind="stable")
    s_ids = ids[order]
    s_other = other[order]

    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(ptr[1:], s_ids, 1)
    np.cumsum(ptr, out=ptr)

    return torch.from_numpy(ptr), torch.from_numpy(s_other)


def build_edge_dict(edge_index):
    users = edge_index[0].numpy().astype(np.int64)
    repos = edge_index[1].numpy().astype(np.int64)
    order = np.argsort(users, kind="stable")
    s_users = users[order]
    s_repos = repos[order]
    splits = np.flatnonzero(np.diff(s_users)) + 1
    unique_u = np.concatenate([[s_users[0]], s_users[splits]])
    repo_groups = np.split(s_repos, splits)
    return {int(u): g.tolist() for u, g in zip(unique_u, repo_groups)}


def compute_user_tag_profiles(train_csr_ptr, train_csr_col, repo_tags, repo_names):
    num_users = train_csr_ptr.size(0) - 1
    user_tag_counts = [defaultdict(float) for _ in range(num_users)]
    
    for u in range(num_users):
        start = train_csr_ptr[u].item()
        end = train_csr_ptr[u + 1].item()
        user_repos = train_csr_col[start:end].tolist()
        
        for repo_idx in user_repos:
            repo_name = repo_names[repo_idx]
            if repo_name in repo_tags:
                for tag in repo_tags[repo_name]:
                    user_tag_counts[u][tag] += 1.0
    
    return user_tag_counts


def build_tag_to_repos_index(repo_tags, repo_names):
    tag_to_repos = defaultdict(list)
    for repo_idx, repo_name in enumerate(repo_names):
        if repo_name in repo_tags:
            for tag in repo_tags[repo_name]:
                tag_to_repos[tag].append(repo_idx)
    return tag_to_repos


def evaluate(user_tag_profiles, repo_tags, repo_names, tag_to_repos,
             eval_dict, eval_users, train_csr_ptr, train_csr_col, k=20):
    num_repos = len(repo_names)
    
    log2_table = np.log2(np.arange(2, k + 2, dtype=np.float32))
    idcg_table = np.cumsum(1.0 / log2_table)
    
    total_recall = 0.0
    total_ndcg = 0.0
    count = 0
    
    for idx, u in enumerate(eval_users):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(eval_users)} users...")
        
        u_idx = u.item() if isinstance(u, torch.Tensor) else u
        
        gt = eval_dict.get(u_idx, [])
        if not gt:
            continue
        
        user_train_repos = set(train_csr_col[train_csr_ptr[u_idx]:train_csr_ptr[u_idx + 1]].tolist())
        user_tags = user_tag_profiles[u_idx]
        
        if not user_tags:
            count += 1
            continue
        
        total_weight = sum(user_tags.values())
        
        candidate_repos = set()
        for tag in user_tags.keys():
            if tag in tag_to_repos:
                candidate_repos.update(tag_to_repos[tag])
        
        candidate_repos -= user_train_repos
        
        if len(candidate_repos) < k:
            count += 1
            continue
        
        scores = []
        repo_indices = []
        for repo_idx in candidate_repos:
            repo_name = repo_names[repo_idx]
            if repo_name not in repo_tags:
                continue
            
            score = 0.0
            for tag in repo_tags[repo_name]:
                if tag in user_tags:
                    score += user_tags[tag] / total_weight
            
            scores.append(score)
            repo_indices.append(repo_idx)
        
        if len(scores) < k:
            count += 1
            continue
        
        scores = np.array(scores, dtype=np.float32)
        repo_indices = np.array(repo_indices, dtype=np.int32)
        
        topk_mask = np.argpartition(scores, -k)[-k:]
        topk_idx = repo_indices[topk_mask[np.argsort(-scores[topk_mask])]]
        
        hits = np.isin(topk_idx, gt).astype(np.float32)
        
        recall = hits.sum() / min(len(gt), k)
        total_recall += recall
        
        dcg = (hits / log2_table).sum()
        ideal_k = min(len(gt), k)
        idcg = idcg_table[ideal_k - 1]
        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
        
        count += 1
    
    return (total_recall / count if count else 0.0, total_ndcg / count if count else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    
    print("Loading tags...")
    t0 = time.time()
    repo_tags = load_tags()
    print(f"  Loaded tags for {len(repo_tags)} repos [{time.time()-t0:.1f}s]")
    
    print("Loading graph metadata...")
    data, meta = load_data()
    repo_names = meta["repo_names"]
    num_users = data["user"].num_nodes
    num_repos = data["repo"].num_nodes
    print(f"  Users: {num_users}, Repos: {num_repos}")
    
    print("Splitting edges...")
    t0 = time.time()
    edge_index = data["user", "stars", "repo"].edge_index
    train_idx, val_idx, test_idx = train_val_test_split(data)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}  [{time.time()-t0:.1f}s]")
    
    train_edge_index = edge_index[:, train_idx]
    eval_edges_val = edge_index[:, val_idx]
    eval_edges_test = edge_index[:, test_idx]
    
    print("Building train CSR...")
    t0 = time.time()
    train_csr_ptr, train_csr_col = build_csr(train_edge_index, num_users, dim=0)
    print(f"  Done [{time.time()-t0:.1f}s]")
    
    print("Building eval dicts...")
    t0 = time.time()
    val_dict = build_edge_dict(eval_edges_val)
    test_dict = build_edge_dict(eval_edges_test)
    combined_eval_dict = {}
    for u, repos in val_dict.items():
        combined_eval_dict[u] = list(repos)
    for u, repos in test_dict.items():
        if u in combined_eval_dict:
            combined_eval_dict[u].extend(repos)
        else:
            combined_eval_dict[u] = list(repos)
    val_users = eval_edges_val[0].unique()
    test_users = eval_edges_test[0].unique()
    print(f"  Val users: {len(val_users)}, Test users: {len(test_users)}  [{time.time()-t0:.1f}s]")
    
    print(f"Computing user tag profiles from training data...")
    t0 = time.time()
    user_tag_profiles = compute_user_tag_profiles(train_csr_ptr, train_csr_col, repo_tags, repo_names)
    print(f"  Done [{time.time()-t0:.1f}s]")
    
    print("Building tag-to-repos inverted index...")
    t0 = time.time()
    tag_to_repos = build_tag_to_repos_index(repo_tags, repo_names)
    print(f"  Done [{time.time()-t0:.1f}s]")
    
    print(f"\nEvaluating on validation set (k={args.k})...")
    t0 = time.time()
    val_recall, val_ndcg = evaluate(
        user_tag_profiles, repo_tags, repo_names, tag_to_repos,
        combined_eval_dict, val_users, train_csr_ptr, train_csr_col, k=args.k
    )
    print(f"  Val Recall@{args.k}: {val_recall:.4f} | NDCG@{args.k}: {val_ndcg:.4f}  [{time.time()-t0:.1f}s]")
    
    print(f"\nEvaluating on test set (k={args.k})...")
    t0 = time.time()
    test_recall, test_ndcg = evaluate(
        user_tag_profiles, repo_tags, repo_names, tag_to_repos,
        combined_eval_dict, test_users, train_csr_ptr, train_csr_col, k=args.k
    )
    print(f"  Test Recall@{args.k}: {test_recall:.4f} | NDCG@{args.k}: {test_ndcg:.4f}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
