import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from model_simple import RepoEmbeddingModel

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def load_data():
    data = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
    return data, meta


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


def sample_costar_batch(csr_ptr, csr_col, eligible_users, batch_size, num_neg, num_repos, device):
    """Sample positive co-star pairs and negative repos using the user CSR."""
    idx = torch.randint(0, len(eligible_users), (batch_size,), device=device)
    users = eligible_users[idx]

    starts = csr_ptr[users]
    ends = csr_ptr[users + 1]
    lengths = ends - starts

    off_a = (torch.rand(batch_size, device=device) * lengths.float()).long()
    off_b = (torch.rand(batch_size, device=device) * (lengths - 1).float()).long()
    off_b = off_b + (off_b >= off_a).long()

    repo_a = csr_col[starts + off_a]
    repo_b = csr_col[starts + off_b]

    neg_repos = torch.randint(0, num_repos, (batch_size, num_neg), device=device)

    return repo_a, repo_b, neg_repos


def train_epoch(model, x_nomic, x_stars, csr_ptr, csr_col, eligible_users,
                optimizer, steps_per_epoch, batch_size, num_neg, num_repos):
    model.train()
    device = x_nomic.device
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        repo_a, repo_b, neg_repos = sample_costar_batch(
            csr_ptr, csr_col, eligible_users, batch_size, num_neg, num_repos, device
        )

        all_repo_embs = model(x_nomic, x_stars)

        emb_a = all_repo_embs[repo_a]
        emb_b = all_repo_embs[repo_b]
        emb_neg = all_repo_embs[neg_repos.view(-1)].view(batch_size, num_neg, -1)

        pos_scores = (emb_a * emb_b).sum(dim=-1)
        neg_scores = (emb_a.unsqueeze(1) * emb_neg).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


@torch.no_grad()
def evaluate(model, x_nomic, x_stars, eval_dict, eval_users,
             train_csr_ptr, train_csr_col, k=20, batch_size=8192):
    model.eval()
    device = x_nomic.device

    repo_embs = model(x_nomic, x_stars)
    num_repos = repo_embs.size(0)
    num_users = train_csr_ptr.size(0) - 1
    d = repo_embs.size(1)

    counts = (train_csr_ptr[1:] - train_csr_ptr[:-1])
    total_edges = int(train_csr_col.size(0))
    edge_repos = train_csr_col[:total_edges]
    edge_users = torch.arange(num_users, device=device).repeat_interleave(counts)

    user_embs = torch.zeros(num_users, d, device=device)
    user_embs.scatter_add_(0, edge_users.unsqueeze(1).expand(-1, d), repo_embs[edge_repos])
    user_embs = user_embs / counts.float().clamp(min=1).unsqueeze(1)
    user_embs = F.normalize(user_embs, dim=-1)

    log2_table = torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=device))
    idcg_table = (1.0 / log2_table).cumsum(0)

    total_recall = 0.0
    total_ndcg = 0.0
    count = 0

    for start in range(0, len(eval_users), batch_size):
        batch_u = eval_users[start:start + batch_size].to(device)
        B = len(batch_u)

        u_embs = user_embs[batch_u]
        scores = u_embs @ repo_embs.t()

        starts_csr = train_csr_ptr[batch_u]
        ends_csr = train_csr_ptr[batch_u + 1]
        lengths = ends_csr - starts_csr
        max_len = int(lengths.max().item())
        if max_len > 0:
            pad_offsets = torch.arange(max_len, device=device).unsqueeze(0)
            padded_idx = (starts_csr.unsqueeze(1) + pad_offsets).clamp(0, train_csr_col.size(0) - 1)
            valid = pad_offsets < lengths.unsqueeze(1)
            repo_ids = train_csr_col[padded_idx]
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_len)[valid]
            col_idx = repo_ids[valid]
            scores[row_idx, col_idx] = -1e9

        _, topk_idx = scores.topk(k, dim=1)

        gt_rows, gt_repos_flat, gt_counts_list = [], [], []
        batch_u_list = batch_u.tolist()
        for i, u in enumerate(batch_u_list):
            gt = eval_dict.get(u, [])
            gt_counts_list.append(len(gt))
            gt_rows.extend([i] * len(gt))
            gt_repos_flat.extend(gt)

        gt_counts = torch.tensor(gt_counts_list, dtype=torch.float, device=device)
        has_gt = gt_counts > 0
        if not has_gt.any():
            continue

        if gt_rows:
            gt_encoded = (
                torch.tensor(gt_rows, device=device) * num_repos
                + torch.tensor(gt_repos_flat, device=device)
            )
            topk_encoded = torch.arange(B, device=device).unsqueeze(1) * num_repos + topk_idx
            hits = torch.isin(topk_encoded.view(-1), gt_encoded).view(B, k).float()
        else:
            hits = torch.zeros(B, k, device=device)

        recall_per = hits.sum(dim=1) / gt_counts.clamp(min=1, max=k)
        total_recall += recall_per[has_gt].sum().item()

        dcg = (hits / log2_table).sum(dim=1)
        ideal_k = gt_counts.long().clamp(max=k)
        idcg = torch.zeros(B, device=device)
        idcg[has_gt] = idcg_table[ideal_k[has_gt] - 1]
        total_ndcg += (dcg / idcg.clamp(min=1e-9))[has_gt].sum().item()
        count += int(has_gt.sum().item())

    return (total_recall / count if count else 0.0, total_ndcg / count if count else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_neg", type=int, default=9)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading data...")
    data, meta = load_data()
    num_users = data["user"].num_nodes
    num_repos = data["repo"].num_nodes
    print(f"  Users: {num_users}, Repos: {num_repos}")
    print(f"  Edges: {data['user', 'stars', 'repo'].edge_index.size(1)}")

    print("Splitting edges...")
    t0 = time.time()
    train_idx, val_idx, test_idx = train_val_test_split(data)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}  [{time.time()-t0:.1f}s]")

    edge_index = data["user", "stars", "repo"].edge_index
    train_edge_index = edge_index[:, train_idx]
    eval_edges_val = edge_index[:, val_idx]
    eval_edges_test = edge_index[:, test_idx]

    print("Building train CSR...")
    t0 = time.time()
    train_csr_ptr, train_csr_col = build_csr(train_edge_index, num_users, dim=0)
    train_csr_ptr = train_csr_ptr.to(args.device)
    train_csr_col = train_csr_col.to(args.device)
    print(f"  Done [{time.time()-t0:.1f}s]")

    lengths = train_csr_ptr[1:] - train_csr_ptr[:-1]
    eligible_users = torch.where(lengths >= 2)[0].to(args.device)
    print(f"  Users with >=2 train stars (eligible for co-star sampling): {len(eligible_users)}")

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

    x_nomic = data["repo"].x_nomic.to(args.device)
    x_stars = data["repo"].x_stars.to(args.device)

    model = RepoEmbeddingModel(
        nomic_dim=768, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
    ).to(args.device)

    with torch.no_grad():
        model(x_nomic, x_stars)

    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_recall = 0.0
    patience_counter = 0
    patience = 10
    ckpt_path = os.path.join(DATA_DIR, "best_model_simple.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, x_nomic, x_stars, train_csr_ptr, train_csr_col,
            eligible_users, optimizer,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            num_neg=args.num_neg,
            num_repos=num_repos,
        )
        scheduler.step()
        t_train = time.time() - t0

        log = f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train: {t_train:.1f}s"

        if (epoch % args.eval_every == 0) or (epoch == 1):
            t1 = time.time()
            val_recall, val_ndcg = evaluate(
                model, x_nomic, x_stars, combined_eval_dict, val_users,
                train_csr_ptr, train_csr_col,
                k=args.k, batch_size=args.eval_batch_size,
            )
            t_val = time.time() - t1
            log += f" | Val R@{args.k}: {val_recall:.4f} | NDCG: {val_ndcg:.4f} | Eval: {t_val:.1f}s"
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
                log += " *"
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(log)
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(log)

    print(f"\nBest Val Recall@{args.k}: {best_val_recall:.4f}")
    print("Loading best model for test evaluation...")
    state = torch.load(ckpt_path, weights_only=True)
    fixed = {}
    for k, v in state.items():
        clean = k.replace("_orig_mod.", "")
        fixed[clean] = v
        fixed[f"_orig_mod.{clean}"] = v
    model.load_state_dict(fixed, strict=False)
    test_recall, test_ndcg = evaluate(
        model, x_nomic, x_stars, combined_eval_dict, test_users,
        train_csr_ptr, train_csr_col,
        k=args.k, batch_size=args.eval_batch_size,
    )
    print(f"Test Recall@{args.k}: {test_recall:.4f} | Test NDCG@{args.k}: {test_ndcg:.4f}")


if __name__ == "__main__":
    main()
