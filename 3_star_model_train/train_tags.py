import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from model_tags import TagEmbeddingModel

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS_DIR = os.path.join(BASE_DIR, "1_3_repo_tags")
REPOS_CSV = os.path.join(BASE_DIR, "1_2_collect_api", "github_repos_100plus_stars.csv")


def load_data():
    data = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)
    return data, meta


def load_tags():
    repos_df = pd.read_csv(REPOS_CSV)
    row_id_to_name = {row['repo_id']: row['repo_name'] for _, row in repos_df.iterrows()}
    del repos_df
    gc.collect()

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
                    repo_tags[repo_name] = list(data["tags"])
    del row_id_to_name
    gc.collect()
    return repo_tags


def build_tag_vocab(repo_tags):
    all_tags = set()
    for tags in repo_tags.values():
        all_tags.update(tags)
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    return tag2idx


def build_repo_tag_tensors(repo_names, repo_tags, tag2idx, max_tags=20):
    num_repos = len(repo_names)
    tag_indices = torch.zeros(num_repos, max_tags, dtype=torch.int32)
    tag_weights = torch.zeros(num_repos, max_tags, dtype=torch.float16)

    for repo_idx, repo_name in enumerate(repo_names):
        if repo_name in repo_tags:
            tags = repo_tags[repo_name][:max_tags]
            for i, tag in enumerate(tags):
                if tag in tag2idx:
                    tag_indices[repo_idx, i] = tag2idx[tag]
                    tag_weights[repo_idx, i] = 1.0

    return tag_indices, tag_weights


def split_users(num_users, edge_index, min_stars=2, seed=42, train_frac=0.9, val_frac=0.05):
    rng = np.random.default_rng(seed)
    users = edge_index[0].numpy().astype(np.int64)
    unique, counts = np.unique(users, return_counts=True)
    eligible = unique[counts >= min_stars]
    rng.shuffle(eligible)
    n = len(eligible)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    return (
        torch.from_numpy(np.sort(eligible[:n_train])),
        torch.from_numpy(np.sort(eligible[n_train:n_val])),
        torch.from_numpy(np.sort(eligible[n_val:])),
    )


def build_csr(edge_index, num_nodes, dim=0):
    ids = edge_index[dim].numpy().astype(np.int32)
    other = edge_index[1 - dim].numpy().astype(np.int32)
    order = np.argsort(ids, kind="stable")
    s_ids = ids[order]
    s_other = other[order]

    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(ptr[1:], s_ids, 1)
    np.cumsum(ptr, out=ptr)

    return torch.from_numpy(ptr), torch.from_numpy(s_other.astype(np.int64))


def _sample_negatives_excluding_positives(users, csr_ptr, csr_col, num_neg, num_repos, device, max_resample=8):
    B = users.size(0)
    starts = csr_ptr[users]
    ends = csr_ptr[users + 1]
    lengths = ends - starts
    max_len = int(lengths.max().item())

    neg_repos = torch.randint(0, num_repos, (B, num_neg), device=device)
    if max_len == 0:
        return neg_repos, 0.0

    pad_offsets = torch.arange(max_len, device=device).unsqueeze(0)
    padded_idx = (starts.unsqueeze(1) + pad_offsets).clamp(0, csr_col.size(0) - 1)
    pos_repos = csr_col[padded_idx]
    valid = pad_offsets < lengths.unsqueeze(1)

    initial_pos = ((neg_repos.unsqueeze(-1) == pos_repos.unsqueeze(1)) & valid.unsqueeze(1)).any(dim=-1)
    false_neg_rate = initial_pos.float().mean().item()
    is_pos = initial_pos

    for _ in range(max_resample):
        if not is_pos.any():
            break
        neg_repos[is_pos] = torch.randint(0, num_repos, (int(is_pos.sum().item()),), device=device)
        is_pos = ((neg_repos.unsqueeze(-1) == pos_repos.unsqueeze(1)) & valid.unsqueeze(1)).any(dim=-1)

    return neg_repos, false_neg_rate


def sample_costar_batch(csr_ptr, csr_col, eligible_users, batch_size, num_neg, num_repos, device):
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

    neg_repos, false_neg_rate = _sample_negatives_excluding_positives(
        users, csr_ptr, csr_col, num_neg, num_repos, device
    )

    return users, repo_a, repo_b, neg_repos, false_neg_rate


def train_epoch(model, tag_indices, tag_weights, csr_ptr, csr_col, eligible_users,
                optimizer, steps_per_epoch, batch_size, num_neg, num_repos,
                bpr_margin=0.0, bpr_temp=1.0, inbatch_neg_weight=0.0,
                inbatch_temp=0.2, hard_neg_ratio=0.0, hard_neg_pool_mult=4):
    model.train()
    device = tag_indices.device
    total_loss = 0.0
    total_false_neg = 0.0
    pos_stats = None
    neg_stats = None

    hard_neg_ratio = float(max(0.0, min(1.0, hard_neg_ratio)))
    hard_neg_k = int(round(num_neg * hard_neg_ratio))
    hard_neg_k = min(max(hard_neg_k, 0), num_neg)

    for _ in range(steps_per_epoch):
        users, repo_a, repo_b, neg_repos, false_neg_rate = sample_costar_batch(
            csr_ptr, csr_col, eligible_users, batch_size, num_neg, num_repos, device
        )
        total_false_neg += false_neg_rate

        all_repo_embs = model(tag_indices, tag_weights)

        emb_a = all_repo_embs[repo_a]
        emb_b = all_repo_embs[repo_b]

        if hard_neg_k > 0:
            hard_pool = max(hard_neg_k * hard_neg_pool_mult, hard_neg_k)
            hard_pool_repos, _ = _sample_negatives_excluding_positives(
                users, csr_ptr, csr_col, hard_pool, num_repos, device
            )
            hard_pool_embs = all_repo_embs[hard_pool_repos.view(-1)].view(batch_size, hard_pool, -1)
            hard_scores = (emb_a.unsqueeze(1) * hard_pool_embs).sum(dim=-1)
            hard_idx = hard_scores.topk(hard_neg_k, dim=1).indices
            neg_repos[:, :hard_neg_k] = hard_pool_repos.gather(1, hard_idx)

        emb_neg = all_repo_embs[neg_repos.view(-1)].view(batch_size, num_neg, -1)

        pos_scores = (emb_a * emb_b).sum(dim=-1)
        neg_scores = (emb_a.unsqueeze(1) * emb_neg).sum(dim=-1)
        bpr_logits = (pos_scores.unsqueeze(1) - neg_scores - bpr_margin) / max(bpr_temp, 1e-6)
        bpr_loss = -F.logsigmoid(bpr_logits).mean()

        if inbatch_neg_weight > 0.0:
            sim = emb_a @ emb_b.t()
            labels = torch.arange(batch_size, device=device)
            inbatch_loss = F.cross_entropy(sim / max(inbatch_temp, 1e-6), labels)
            loss = bpr_loss + inbatch_neg_weight * inbatch_loss
        else:
            loss = bpr_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        pos_stats = torch.quantile(pos_scores.detach(), torch.tensor([0.5, 0.9, 0.99], device=device)).cpu()
        neg_stats = torch.quantile(neg_scores.detach().reshape(-1), torch.tensor([0.5, 0.9, 0.99], device=device)).cpu()

    stats = {
        "false_neg_rate": total_false_neg / steps_per_epoch,
        "pos_q": pos_stats.tolist() if pos_stats is not None else [0.0, 0.0, 0.0],
        "neg_q": neg_stats.tolist() if neg_stats is not None else [0.0, 0.0, 0.0],
    }
    return total_loss / steps_per_epoch, stats


@torch.no_grad()
def evaluate(model, tag_indices, tag_weights, csr_ptr, csr_col, eval_users,
             num_repos, k=20, batch_size=4096):
    """Leave-one-out evaluation: hold out 1 random star, build embedding from the rest."""
    model.eval()
    device = tag_indices.device

    repo_embs = model(tag_indices, tag_weights)
    d = repo_embs.size(1)
    num_users = csr_ptr.size(0) - 1

    counts = csr_ptr[1:] - csr_ptr[:-1]
    edge_repos = csr_col[:int(csr_col.size(0))]
    edge_users_idx = torch.arange(num_users, device=device).repeat_interleave(counts)
    user_sums = torch.zeros(num_users, d, device=device)
    user_sums.scatter_add_(0, edge_users_idx.unsqueeze(1).expand(-1, d), repo_embs[edge_repos])
    del edge_repos, edge_users_idx

    log2_table = torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=device))

    total_recall = 0.0
    total_ndcg = 0.0
    count = 0

    for start in range(0, len(eval_users), batch_size):
        batch_u = eval_users[start:start + batch_size].to(device)
        B = len(batch_u)

        u_counts = counts[batch_u]
        has_enough = u_counts >= 2
        if not has_enough.any():
            continue
        batch_u = batch_u[has_enough]
        B = len(batch_u)
        u_counts = counts[batch_u]

        starts_csr = csr_ptr[batch_u]
        lengths = u_counts

        held_out_off = (torch.rand(B, device=device) * lengths.float()).long()
        held_out_repos = csr_col[starts_csr + held_out_off]
        held_out_embs = repo_embs[held_out_repos]

        loo_embs = (user_sums[batch_u] - held_out_embs) / (u_counts - 1).float().unsqueeze(1).clamp(min=1)
        loo_embs = F.normalize(loo_embs, dim=-1)

        scores = loo_embs @ repo_embs.t()

        max_len = int(lengths.max().item())
        if max_len > 0:
            pad_offsets = torch.arange(max_len, device=device).unsqueeze(0)
            padded_idx = (starts_csr.unsqueeze(1) + pad_offsets).clamp(0, csr_col.size(0) - 1)
            valid = pad_offsets < lengths.unsqueeze(1)
            repo_ids = csr_col[padded_idx]
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_len)[valid]
            col_idx = repo_ids[valid]
            scores[row_idx, col_idx] = -1e9

        scores[torch.arange(B, device=device), held_out_repos] = (loo_embs * held_out_embs).sum(dim=-1)

        _, topk_idx = scores.topk(k, dim=1)
        del scores

        hits = (topk_idx == held_out_repos.unsqueeze(1)).float()
        total_recall += hits.sum().item()
        dcg = (hits / log2_table).sum(dim=1)
        total_ndcg += dcg.sum().item()
        count += B

    return total_recall, total_ndcg, count


@torch.no_grad()
def eval_bpr_loss(repo_embs, csr_ptr, csr_col, users, num_repos, num_neg=9, batch_size=8192):
    """BPR loss on co-star pairs from the given users."""
    device = repo_embs.device
    total_loss = 0.0
    total_count = 0

    for start in range(0, len(users), batch_size):
        batch_u = users[start:start + batch_size].to(device)

        s = csr_ptr[batch_u]
        l = csr_ptr[batch_u + 1] - s
        valid = l >= 2
        if not valid.any():
            continue

        bu = batch_u[valid]
        Bv = int(valid.sum().item())
        s = csr_ptr[bu]
        l = csr_ptr[bu + 1] - s

        off_a = (torch.rand(Bv, device=device) * l.float()).long()
        off_b = (torch.rand(Bv, device=device) * (l - 1).float()).long()
        off_b = off_b + (off_b >= off_a).long()

        repo_a = csr_col[s + off_a]
        repo_b = csr_col[s + off_b]
        neg_repos = torch.randint(0, num_repos, (Bv, num_neg), device=device)

        emb_a = repo_embs[repo_a]
        emb_b = repo_embs[repo_b]
        emb_neg = repo_embs[neg_repos.view(-1)].view(Bv, num_neg, -1)

        pos_scores = (emb_a * emb_b).sum(dim=-1)
        neg_scores = (emb_a.unsqueeze(1) * emb_neg).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

        total_loss += loss.item() * Bv
        total_count += Bv

    return total_loss, total_count


def preprocess_and_save(path):
    """Rank 0 loads raw data, preprocesses, saves compact tensors to a temp file."""
    print("Loading data...")
    data, meta = load_data()
    repo_names = meta["repo_names"]
    num_users = data["user"].num_nodes
    num_repos = data["repo"].num_nodes
    print(f"  Users: {num_users}, Repos: {num_repos}")

    print("Loading tags...")
    repo_tags = load_tags()
    print(f"  Loaded tags for {len(repo_tags)} repos")

    print("Building tag vocabulary...")
    tag2idx = build_tag_vocab(repo_tags)
    num_tags = len(tag2idx)
    print(f"  Vocabulary size: {num_tags} unique tags")

    print("Building repo tag tensors...")
    tag_indices, tag_weights = build_repo_tag_tensors(repo_names, repo_tags, tag2idx, max_tags=20)
    del repo_tags, tag2idx, repo_names
    gc.collect()

    edge_index = data["user", "stars", "repo"].edge_index
    del data, meta
    gc.collect()

    print("Splitting users (90/5/5)...")
    t0 = time.time()
    train_users, val_users, test_users = split_users(num_users, edge_index)
    print(f"  Train users: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}  [{time.time()-t0:.1f}s]")

    print("Building CSR...")
    t0 = time.time()
    csr_ptr, csr_col = build_csr(edge_index, num_users, dim=0)
    del edge_index
    gc.collect()
    print(f"  Done [{time.time()-t0:.1f}s]")

    torch.save({
        "num_users": num_users,
        "num_repos": num_repos,
        "num_tags": num_tags,
        "tag_indices": tag_indices,
        "tag_weights": tag_weights,
        "csr_ptr": csr_ptr,
        "csr_col": csr_col,
        "train_users": train_users,
        "val_users": val_users,
        "test_users": test_users,
    }, path)
    print(f"Saved preprocessed data to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_tags", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_neg", type=int, default=9)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.0)
    parser.add_argument("--hard_neg_pool_mult", type=int, default=4)
    parser.add_argument("--bpr_margin", type=float, default=0.0)
    parser.add_argument("--bpr_temp", type=float, default=1.0)
    parser.add_argument("--inbatch_neg_weight", type=float, default=0.0)
    parser.add_argument("--inbatch_temp", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0:
            print(f"Distributed training: {world_size} GPUs")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(args.device)

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    prep_path = os.path.join(DATA_DIR, "_prep_tags_tmp.pt")
    if rank == 0:
        preprocess_and_save(prep_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("All ranks loading preprocessed data...")
    prep = torch.load(prep_path, weights_only=False)

    num_users = prep["num_users"]
    num_repos = prep["num_repos"]
    num_tags = prep["num_tags"]
    tag_indices = prep["tag_indices"].long().to(device)
    tag_weights = prep["tag_weights"].float().to(device)
    csr_ptr = prep["csr_ptr"].to(device)
    csr_col = prep["csr_col"].to(device)
    train_users = prep["train_users"].to(device)
    val_users = prep["val_users"]
    test_users = prep["test_users"]
    del prep
    gc.collect()

    if world_size > 1:
        val_users_local = val_users[rank::world_size]
        test_users_local = test_users[rank::world_size]
    else:
        val_users_local = val_users
        test_users_local = test_users

    num_train_eval = min(int(train_users.size(0)), 50000)
    rng_eval = np.random.default_rng(42)
    train_eval_perm = rng_eval.choice(int(train_users.size(0)), size=num_train_eval, replace=False)
    train_eval_users = train_users.cpu()[torch.from_numpy(train_eval_perm)]
    if world_size > 1:
        train_eval_users_local = train_eval_users[rank::world_size]
    else:
        train_eval_users_local = train_eval_users
    del train_eval_users, val_users, test_users
    gc.collect()

    if rank == 0:
        print(f"Train users: {len(train_users)}, Val eval: {len(val_users_local)}, "
              f"Train eval subsample: {len(train_eval_users_local)}")

    model = TagEmbeddingModel(
        num_tags=num_tags, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    with torch.no_grad():
        model(tag_indices[:100], tag_weights[:100])

    if rank == 0:
        print("Compiling model with torch.compile...")
    model = torch.compile(model)

    model_module = model
    while hasattr(model_module, "module"):
        model_module = model_module.module
    num_params = sum(p.numel() for p in model_module.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_recall = 0.0
    patience_counter = 0
    patience = 10
    ckpt_path = os.path.join(DATA_DIR, "best_model_tags.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_stats = train_epoch(
            model, tag_indices, tag_weights, csr_ptr, csr_col,
            train_users, optimizer,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            num_neg=args.num_neg,
            num_repos=num_repos,
            bpr_margin=args.bpr_margin,
            bpr_temp=args.bpr_temp,
            inbatch_neg_weight=args.inbatch_neg_weight,
            inbatch_temp=args.inbatch_temp,
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_pool_mult=args.hard_neg_pool_mult,
        )

        if world_size > 1:
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / world_size

        scheduler.step()
        t_train = time.time() - t0

        pos_q = train_stats["pos_q"]
        neg_q = train_stats["neg_q"]
        log = f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train: {t_train:.1f}s"
        log += f" | FN%: {100.0 * train_stats['false_neg_rate']:.2f}"
        log += f" | PosQ(50/90/99): {pos_q[0]:.3f}/{pos_q[1]:.3f}/{pos_q[2]:.3f}"
        log += f" | NegQ(50/90/99): {neg_q[0]:.3f}/{neg_q[1]:.3f}/{neg_q[2]:.3f}"

        if (epoch % args.eval_every == 0) or (epoch == 1):
            t1 = time.time()

            model.eval()
            with torch.no_grad():
                repo_embs = model(tag_indices, tag_weights)

            tr_bpr_sum, tr_bpr_n = eval_bpr_loss(
                repo_embs, csr_ptr, csr_col, train_eval_users_local,
                num_repos, num_neg=args.num_neg,
            )
            vl_bpr_sum, vl_bpr_n = eval_bpr_loss(
                repo_embs, csr_ptr, csr_col, val_users_local,
                num_repos, num_neg=args.num_neg,
            )
            del repo_embs

            if world_size > 1:
                bpr_m = torch.tensor([tr_bpr_sum, float(tr_bpr_n), vl_bpr_sum, float(vl_bpr_n)], device=device)
                dist.all_reduce(bpr_m, op=dist.ReduceOp.SUM)
                train_bpr = bpr_m[0].item() / max(bpr_m[1].item(), 1)
                val_bpr = bpr_m[2].item() / max(bpr_m[3].item(), 1)
            else:
                train_bpr = tr_bpr_sum / max(tr_bpr_n, 1)
                val_bpr = vl_bpr_sum / max(vl_bpr_n, 1)

            tr_recall_sum, tr_ndcg_sum, tr_count = evaluate(
                model, tag_indices, tag_weights, csr_ptr, csr_col, train_eval_users_local,
                num_repos, k=args.k, batch_size=args.eval_batch_size,
            )
            vl_recall_sum, vl_ndcg_sum, vl_count = evaluate(
                model, tag_indices, tag_weights, csr_ptr, csr_col, val_users_local,
                num_repos, k=args.k, batch_size=args.eval_batch_size,
            )

            if world_size > 1:
                m = torch.tensor([
                    tr_recall_sum, tr_ndcg_sum, float(tr_count),
                    vl_recall_sum, vl_ndcg_sum, float(vl_count),
                ], device=device)
                dist.all_reduce(m, op=dist.ReduceOp.SUM)
                tr_recall = m[0].item() / max(m[2].item(), 1)
                tr_ndcg = m[1].item() / max(m[2].item(), 1)
                val_recall = m[3].item() / max(m[5].item(), 1)
                val_ndcg = m[4].item() / max(m[5].item(), 1)
            else:
                tr_recall = tr_recall_sum / max(tr_count, 1)
                tr_ndcg = tr_ndcg_sum / max(tr_count, 1)
                val_recall = vl_recall_sum / max(vl_count, 1)
                val_ndcg = vl_ndcg_sum / max(vl_count, 1)

            t_eval = time.time() - t1
            log += f" | T-BPR: {train_bpr:.4f} V-BPR: {val_bpr:.4f}"
            log += f" | T-R@{args.k}: {tr_recall:.4f} V-R@{args.k}: {val_recall:.4f}"
            log += f" | T-NDCG: {tr_ndcg:.4f} V-NDCG: {val_ndcg:.4f}"
            log += f" | Eval: {t_eval:.1f}s"

            if rank == 0:
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    torch.save(model_module.state_dict(), ckpt_path)
                    patience_counter = 0
                    log += " *"
                else:
                    patience_counter += 1

            if world_size > 1:
                patience_tensor = torch.tensor([patience_counter], device=device)
                dist.broadcast(patience_tensor, src=0)
                patience_counter = int(patience_tensor.item())

            if patience_counter >= patience:
                if rank == 0:
                    print(log)
                    print(f"Early stopping at epoch {epoch}")
                break

        if rank == 0:
            print(log)

    if rank == 0:
        print(f"\nBest Val Recall@{args.k}: {best_val_recall:.4f}")
        print("Loading best model for test evaluation...")

    if world_size > 1:
        dist.barrier()
    state = torch.load(ckpt_path, weights_only=True)
    fixed = {}
    for k_name, v in state.items():
        clean = k_name.replace("_orig_mod.", "")
        fixed[clean] = v
    model_module.load_state_dict(fixed, strict=False)
    del state, fixed

    local_test_recall, local_test_ndcg, local_test_count = evaluate(
        model, tag_indices, tag_weights, csr_ptr, csr_col, test_users_local,
        num_repos, k=args.k, batch_size=args.eval_batch_size,
    )

    if world_size > 1:
        metrics = torch.tensor([local_test_recall, local_test_ndcg, float(local_test_count)], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        test_recall = metrics[0].item() / max(metrics[2].item(), 1)
        test_ndcg = metrics[1].item() / max(metrics[2].item(), 1)
    else:
        test_recall = local_test_recall / max(local_test_count, 1)
        test_ndcg = local_test_ndcg / max(local_test_count, 1)

    if rank == 0:
        print(f"Test Recall@{args.k}: {test_recall:.4f} | Test NDCG@{args.k}: {test_ndcg:.4f}")
        if os.path.exists(prep_path):
            os.remove(prep_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
