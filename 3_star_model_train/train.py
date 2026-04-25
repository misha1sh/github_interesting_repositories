import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from model import InductivePinSage

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def load_data():
    data = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
    return data, meta


def train_val_test_split(data, seed=42):
    """Vectorized per-user split. Hold out 1 test edge per user, 1 val if ≥3 edges."""
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


def build_graph_data(data, train_edge_index, device):
    """Build graph tensors using ONLY train edges for message passing."""
    edge_r2u = torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
    return {
        "x_nomic": data["repo"].x_nomic.to(device),
        "x_stars": data["repo"].x_stars.to(device),
        "edge_u2r": train_edge_index.to(device),
        "edge_r2u": edge_r2u.to(device),
        "num_users": data["user"].num_nodes,
        "num_repos": data["repo"].num_nodes,
    }


def build_csr(edge_index, num_users):
    """Build CSR (ptr, col) for fast per-user repo lookup. Both returned on CPU."""
    users = edge_index[0].numpy().astype(np.int64)
    repos = edge_index[1].numpy().astype(np.int64)
    order = np.argsort(users, kind="stable")
    s_repos = repos[order]
    s_users = users[order]

    ptr = np.zeros(num_users + 1, dtype=np.int64)
    np.add.at(ptr[1:], s_users, 1)
    np.cumsum(ptr, out=ptr)

    return torch.from_numpy(ptr), torch.from_numpy(s_repos)


def build_edge_dict(edge_index):
    """Build {user: [repos]} using numpy sort for speed."""
    users = edge_index[0].numpy().astype(np.int64)
    repos = edge_index[1].numpy().astype(np.int64)
    order = np.argsort(users, kind="stable")
    s_users = users[order]
    s_repos = repos[order]
    splits = np.flatnonzero(np.diff(s_users)) + 1
    unique_u = np.concatenate([[s_users[0]], s_users[splits]])
    repo_groups = np.split(s_repos, splits)
    return {int(u): g.tolist() for u, g in zip(unique_u, repo_groups)}


@torch.no_grad()
def evaluate(model, gd, eval_dict, eval_users, train_csr_ptr, train_csr_col, k=20, batch_size=8192):
    """
    eval_dict: {user_id: [repo_ids]} for val or test
    eval_users: 1D tensor of user IDs to evaluate on
    train_csr_ptr, train_csr_col: CSR of training edges for masking (GPU)
    """
    model.eval()
    device = gd["x_nomic"].device

    h_user, h_repo = model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])
    repo_embs = F.normalize(h_repo, dim=-1)  # [num_repos, d]
    num_repos = repo_embs.size(0)

    log2_table = torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=device))
    idcg_table = (1.0 / log2_table).cumsum(0)  # [k]

    total_recall = 0.0
    total_ndcg = 0.0
    count = 0

    for start in range(0, len(eval_users), batch_size):
        batch_u = eval_users[start : start + batch_size].to(device)
        B = len(batch_u)

        user_embs = F.normalize(h_user[batch_u], dim=-1)
        scores = user_embs @ repo_embs.t()  # [B, num_repos]

        # --- Vectorized train masking via CSR ---
        starts = train_csr_ptr[batch_u]       # [B]
        ends   = train_csr_ptr[batch_u + 1]   # [B]
        lengths = ends - starts               # [B]
        max_len = int(lengths.max().item())
        if max_len > 0:
            # Build padded index tensor [B, max_len], clamp to avoid OOB
            pad_offsets = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
            padded_idx = (starts.unsqueeze(1) + pad_offsets).clamp(0, train_csr_col.size(0) - 1)  # [B, max_len]
            valid = pad_offsets < lengths.unsqueeze(1)  # [B, max_len]
            repo_ids = train_csr_col[padded_idx]        # [B, max_len]
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_len)[valid]
            col_idx = repo_ids[valid]
            scores[row_idx, col_idx] = -1e9

        _, topk_idx = scores.topk(k, dim=1)  # [B, k]

        # --- Vectorized hits via isin ---
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


def train_epoch(model, gd, train_edge_index, optimizer, steps_per_epoch, batch_size, num_neg, edge_dropout=0.0, train_pos_encoded=None):
    model.train()
    device = gd["x_nomic"].device
    num_repos = gd["num_repos"]
    num_train = train_edge_index.size(1)
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        if edge_dropout > 0:
            edge_u2r_aug, _ = dropout_edge(gd["edge_u2r"], p=edge_dropout, training=True)
            edge_r2u_aug, _ = dropout_edge(gd["edge_r2u"], p=edge_dropout, training=True)
        else:
            edge_u2r_aug, edge_r2u_aug = gd["edge_u2r"], gd["edge_r2u"]

        h_user, h_repo = model(gd["x_nomic"], gd["x_stars"], edge_u2r_aug, edge_r2u_aug, gd["num_users"])

        idx = torch.randint(0, num_train, (batch_size,), device=device)
        pos_src = train_edge_index[0, idx]
        pos_dst = train_edge_index[1, idx]

        user_emb = F.normalize(h_user[pos_src], dim=-1)
        pos_emb = F.normalize(h_repo[pos_dst], dim=-1)
        pos_scores = (user_emb * pos_emb).sum(dim=-1)

        neg_dst = torch.randint(0, num_repos, (batch_size, num_neg), device=device)
        if train_pos_encoded is not None:
            neg_encoded = pos_src.unsqueeze(1) * num_repos + neg_dst
            collisions = torch.isin(neg_encoded.view(-1), train_pos_encoded).view(batch_size, num_neg)
            n_coll = collisions.sum().item()
            if n_coll > 0:
                neg_dst[collisions] = torch.randint(0, num_repos, (n_coll,), device=device)

        neg_emb = F.normalize(h_repo[neg_dst.view(-1)].view(batch_size, num_neg, -1), dim=-1)
        neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--conv_type", type=str, default="gat")
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--use_layernorm", action="store_true", default=True)
    parser.add_argument("--normalize_output", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_neg", type=int, default=9)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--edge_dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading data...")
    data, meta = load_data()
    num_users = data["user"].num_nodes
    num_repos = data["repo"].num_nodes
    print(f"  Users: {num_users}, Repos: {num_repos}")
    print(f"  Edges: {data['user', 'stars', 'repo'].edge_index.size(1)}")

    print("Splitting edges (vectorized)...")
    t0 = time.time()
    train_idx, val_idx, test_idx = train_val_test_split(data)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}  [{time.time()-t0:.1f}s]")

    edge_index = data["user", "stars", "repo"].edge_index
    train_edge_index = edge_index[:, train_idx]
    eval_edges_val  = edge_index[:, val_idx]
    eval_edges_test = edge_index[:, test_idx]

    print("Building train-only graph (no leakage)...")
    gd = build_graph_data(data, train_edge_index, args.device)
    train_edge_index_gpu = train_edge_index.to(args.device)

    print("Building train CSR for eval masking...")
    t0 = time.time()
    train_csr_ptr, train_csr_col = build_csr(train_edge_index, num_users)
    train_csr_ptr = train_csr_ptr.to(args.device)
    train_csr_col = train_csr_col.to(args.device)
    print(f"  Done [{time.time()-t0:.1f}s]")

    print("Building encoded positive set for negative filtering...")
    train_pos_encoded = (train_edge_index_gpu[0] * num_repos + train_edge_index_gpu[1])

    print("Building eval dicts...")
    t0 = time.time()
    val_dict  = build_edge_dict(eval_edges_val)
    test_dict = build_edge_dict(eval_edges_test)
    combined_eval_dict = {}
    for u, repos in val_dict.items():
        combined_eval_dict[u] = list(repos)
    for u, repos in test_dict.items():
        if u in combined_eval_dict:
            combined_eval_dict[u].extend(repos)
        else:
            combined_eval_dict[u] = list(repos)
    val_users  = eval_edges_val[0].unique()
    test_users = eval_edges_test[0].unique()
    print(f"  Val users: {len(val_users)}, Test users: {len(test_users)}  [{time.time()-t0:.1f}s]")

    model = InductivePinSage(
        nomic_dim=768, hidden_dim=args.hidden_dim, num_hops=args.num_hops,
        dropout=args.dropout, conv_type=args.conv_type, gat_heads=args.gat_heads,
        use_layernorm=args.use_layernorm, normalize_output=args.normalize_output,
    ).to(args.device)
    with torch.no_grad():
        model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])

    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_recall = 0.0
    patience_counter = 0
    patience = 10
    ckpt_path = os.path.join(DATA_DIR, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, gd, train_edge_index_gpu, optimizer,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            num_neg=args.num_neg,
            edge_dropout=args.edge_dropout,
            train_pos_encoded=train_pos_encoded,
        )
        scheduler.step()
        t_train = time.time() - t0

        log = f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train: {t_train:.1f}s"

        if (epoch % args.eval_every == 0) or (epoch == 1):
            t1 = time.time()
            val_recall, val_ndcg = evaluate(
                model, gd, combined_eval_dict, val_users,
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
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    test_recall, test_ndcg = evaluate(
        model, gd, combined_eval_dict, test_users,
        train_csr_ptr, train_csr_col,
        k=args.k, batch_size=args.eval_batch_size,
    )
    print(f"Test Recall@{args.k}: {test_recall:.4f} | Test NDCG@{args.k}: {test_ndcg:.4f}")


if __name__ == "__main__":
    main()
