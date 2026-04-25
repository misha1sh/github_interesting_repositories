import argparse
import os
import sys
import subprocess
import torch
import torch.nn.functional as F
import optuna
from model import InductivePinSage
from train import load_data, train_val_test_split, build_graph_data, build_csr, build_edge_dict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
K = 20
MAX_EPOCHS = 40
PATIENCE = 6
TUNE_VAL_USERS = 30_000  # sample for fast eval per trial


def worker(gpu_id: int, n_trials: int):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading data...")

    data, meta = load_data()
    train_idx, val_idx, _ = train_val_test_split(data)
    edge_index = data["user", "stars", "repo"].edge_index
    train_edge_index = edge_index[:, train_idx]
    num_repos = data["repo"].num_nodes
    num_users = data["user"].num_nodes

    gd = build_graph_data(data, train_edge_index, device)
    train_edge_index_gpu = train_edge_index.to(device)

    print(f"[GPU {gpu_id}] Building CSR for masking...")
    train_csr_ptr, train_csr_col = build_csr(train_edge_index, num_users)
    train_csr_ptr = train_csr_ptr.to(device)
    train_csr_col = train_csr_col.to(device)

    val_edge_index = edge_index[:, val_idx]
    val_dict = build_edge_dict(val_edge_index)
    all_val_users = val_edge_index[0].unique()
    gen = torch.Generator().manual_seed(42 + gpu_id)
    if len(all_val_users) > TUNE_VAL_USERS:
        perm = torch.randperm(len(all_val_users), generator=gen)[:TUNE_VAL_USERS]
        val_users = all_val_users[perm]
    else:
        val_users = all_val_users
    print(f"[GPU {gpu_id}] Using {len(val_users)} val users for tuning.")

    log2_table = torch.log2(torch.arange(2, K + 2, dtype=torch.float, device=device))
    idcg_table = (1.0 / log2_table).cumsum(0)

    @torch.no_grad()
    def fast_evaluate(model):
        model.eval()
        h_user, h_repo = model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])
        repo_embs = F.normalize(h_repo, dim=-1)

        total_recall = 0.0
        count = 0
        eval_batch = 4096
        for start in range(0, len(val_users), eval_batch):
            batch_u = val_users[start : start + eval_batch].to(device)
            B = len(batch_u)

            user_embs = F.normalize(h_user[batch_u], dim=-1)
            scores = user_embs @ repo_embs.t()

            # Vectorized CSR masking
            starts_csr = train_csr_ptr[batch_u]
            ends_csr   = train_csr_ptr[batch_u + 1]
            lengths    = ends_csr - starts_csr
            max_len    = int(lengths.max().item())
            if max_len > 0:
                pad_offsets = torch.arange(max_len, device=device).unsqueeze(0)
                padded_idx  = (starts_csr.unsqueeze(1) + pad_offsets).clamp(0, train_csr_col.size(0) - 1)
                valid       = pad_offsets < lengths.unsqueeze(1)
                repo_ids    = train_csr_col[padded_idx]
                row_idx     = torch.arange(B, device=device).unsqueeze(1).expand(B, max_len)[valid]
                col_idx     = repo_ids[valid]
                scores[row_idx, col_idx] = -1e9

            _, topk_idx = scores.topk(K, dim=1)

            gt_rows, gt_repos_flat, gt_counts_list = [], [], []
            for i, u in enumerate(batch_u.tolist()):
                gt = val_dict.get(u, [])
                gt_counts_list.append(len(gt))
                gt_rows.extend([i] * len(gt))
                gt_repos_flat.extend(gt)

            gt_counts = torch.tensor(gt_counts_list, dtype=torch.float, device=device)
            has_gt = gt_counts > 0
            if not has_gt.any():
                continue

            if gt_rows:
                gt_enc   = torch.tensor(gt_rows, device=device) * num_repos + torch.tensor(gt_repos_flat, device=device)
                topk_enc = torch.arange(B, device=device).unsqueeze(1) * num_repos + topk_idx
                hits = torch.isin(topk_enc.view(-1), gt_enc).view(B, K).float()
            else:
                hits = torch.zeros(B, K, device=device)

            recall_per = hits.sum(dim=1) / gt_counts.clamp(min=1, max=K)
            total_recall += recall_per[has_gt].sum().item()
            count += int(has_gt.sum().item())

        return total_recall / count if count else 0.0

    def train_step(model, optimizer, batch_size, num_neg):
        model.train()
        num_train = train_edge_index_gpu.size(1)
        h_user, h_repo = model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])

        idx = torch.randint(0, num_train, (batch_size,), device=device)
        user_emb = h_user[train_edge_index_gpu[0, idx]]
        pos_emb  = h_repo[train_edge_index_gpu[1, idx]]
        pos_scores = (user_emb * pos_emb).sum(dim=-1)

        neg_dst  = torch.randint(0, num_repos, (batch_size, num_neg), device=device)
        neg_emb  = h_repo[neg_dst.view(-1)].view(batch_size, num_neg, -1)
        neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def objective(trial: optuna.Trial) -> float:
        hidden_dim  = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        num_hops    = trial.suggest_int("num_hops", 1, 4)
        dropout     = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        conv_type   = trial.suggest_categorical("conv_type", ["sage", "gat"])
        gat_heads   = trial.suggest_categorical("gat_heads", [1, 2, 4, 8]) if conv_type == "gat" else 1
        use_layernorm    = trial.suggest_categorical("use_layernorm", [True, False])
        normalize_output = trial.suggest_categorical("normalize_output", [True, False])
        lr           = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
        num_neg      = trial.suggest_int("num_neg", 1, 15)
        steps_per_epoch = trial.suggest_categorical("steps_per_epoch", [20, 40, 80])

        if conv_type == "gat" and hidden_dim % gat_heads != 0:
            raise optuna.TrialPruned()

        model = InductivePinSage(
            nomic_dim=768, hidden_dim=hidden_dim, num_hops=num_hops,
            dropout=dropout, conv_type=conv_type, gat_heads=gat_heads,
            use_layernorm=use_layernorm, normalize_output=normalize_output,
        ).to(device)
        with torch.no_grad():
            model(gd["x_nomic"], gd["x_stars"], gd["edge_u2r"], gd["edge_r2u"], gd["num_users"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

        best_recall = 0.0
        no_improve = 0
        for epoch in range(1, MAX_EPOCHS + 1):
            for _ in range(steps_per_epoch):
                train_step(model, optimizer, batch_size, num_neg)
            scheduler.step()

            if epoch % 3 == 0 or epoch == 1:
                val_recall = fast_evaluate(model)
                trial.report(val_recall, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if val_recall > best_recall:
                    best_recall = val_recall
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        break

        return best_recall

    print(f"[GPU {gpu_id}] Starting {n_trials} trials...")
    storage = f"sqlite:///{os.path.join(DATA_DIR, 'optuna_v3.db')}"
    study = optuna.create_study(
        direction="maximize",
        study_name="pinsage_v3",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=4),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"[GPU {gpu_id}] Done. Best recall: {study.best_value:.4f}")


def launch_all(n_trials_per_gpu: int, n_gpus: int):
    script = os.path.abspath(__file__)
    procs = []
    for gpu_id in range(n_gpus):
        cmd = [sys.executable, script, "--gpu", str(gpu_id), "--n_trials", str(n_trials_per_gpu)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append((gpu_id, p))
        print(f"Launched worker on GPU {gpu_id} (PID {p.pid})")

    for gpu_id, p in procs:
        stdout, _ = p.communicate()
        lines = stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  [GPU {gpu_id}] {line}")
        print(f"  [GPU {gpu_id}] exit code: {p.returncode}")

    storage = f"sqlite:///{os.path.join(DATA_DIR, 'optuna_v3.db')}"
    study = optuna.load_study(study_name="pinsage_v3", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*70}")
    print(f"RESULTS ({len(completed)} completed trials)")
    print(f"{'='*70}")
    print(f"Best Val Recall@{K}: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nTop 5 trials:")
    for i, t in enumerate(sorted(completed, key=lambda t: t.value, reverse=True)[:5]):
        print(f"  #{i+1} R@{K}={t.value:.4f}  {t.params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_gpus", type=int, default=8)
    args = parser.parse_args()

    if args.gpu >= 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        worker(args.gpu, args.n_trials)
    else:
        launch_all(args.n_trials, args.n_gpus)
