import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────

READMES_DIR = Path("/home/misha-sh/github/1_2_collect_api/readmes")
OUT_DIR = Path("/home/misha-sh/github/1_3_repo_tags")
TAG_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NUM_GPUS = 8  # Set this to 4 if you only want to use 4

# ── Worker Logic (Runs on specific GPU) ───────────────────────────

def run_worker(gpu_id, total_gpus):
    # Imports inside function to avoid heavy loading in manager
    from vllm import LLM, SamplingParams
    
    # Prefix for logs
    log_prefix = f"[GPU {gpu_id}]"
    print(f"{log_prefix} Starting up... (Visible Device: {os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # 1. Load Data (Sharded)
    # ----------------------
    all_files = os.listdir(READMES_DIR)
    files = sorted([f for f in all_files if f.isdigit()], key=lambda x: int(x))
    
    # Distribute files: Worker 0 takes 0, 8, 16... Worker 1 takes 1, 9, 17...
    my_files = files[gpu_id::total_gpus]
    
    repo_ids = []
    texts = []
    
    print(f"{log_prefix} Loading {len(my_files)} files...")
    for fname in my_files:
        path = READMES_DIR / fname
        try:
            # Quick read
            text = path.read_text(errors="replace", encoding="utf-8").strip()
            if text:
                repo_ids.append(int(fname))
                texts.append(text)
        except: continue

    # 2. Check Progress
    # -----------------
    shard_out_path = OUT_DIR / f"tags_part_{gpu_id}.jsonl"
    done_ids = set()
    if shard_out_path.exists():
        with open(shard_out_path, "r") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["repo_id"])
                except: pass
    
    # Filter pending
    pending_indices = [i for i, rid in enumerate(repo_ids) if rid not in done_ids]
    
    if not pending_indices:
        print(f"{log_prefix} Work complete! Exiting.")
        return

    print(f"{log_prefix} Processing {len(pending_indices)} pending files.")

    # 3. Initialize vLLM
    # ------------------
    # Because we used CUDA_VISIBLE_DEVICES in the manager, 
    # vLLM thinks there is only 1 GPU on the whole machine.
    llm = LLM(
        model=TAG_MODEL,
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.92,
        max_model_len=32000, 
        trust_remote_code=True,
        enforce_eager=False, 
        disable_log_stats=True # reduce noise
    )
    
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=128)

    # 4. Processing Loop
    # ------------------
    CHUNK_SIZE = 1000
    chunks = [pending_indices[i:i + CHUNK_SIZE] for i in range(0, len(pending_indices), CHUNK_SIZE)]
    
    SYSTEM_PROMPT = 'You categorize GitHub repositories by their README. Output a JSON array of 3-7 lowercase tags. Example: ["python", "web"].'

    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        prompts = []
        meta = []
        
        for idx in chunk:
            truncated = texts[idx][:12000]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": truncated}
            ]
            # Use chat template
            p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
            meta.append(repo_ids[idx])
        
        # Generate
        outputs = llm.generate(prompts, sampling)
        
        # Save
        with open(shard_out_path, "a") as f:
            for rid, out in zip(meta, outputs):
                raw = out.outputs[0].text.strip()
                # Clean up markdown
                if raw.startswith("```json"): raw = raw[7:]
                if raw.endswith("```"): raw = raw[:-3]
                
                # Sanity check JSON
                try:
                    json.loads(raw)
                    json_str = raw
                except:
                    # make it a valid string array at least
                    json_str = json.dumps([raw])

                f.write(json.dumps({"repo_id": rid, "tags": json.loads(json_str)}) + "\n")
        
        elapsed = time.time() - start_time
        avg = ((i + 1) * CHUNK_SIZE) / elapsed
        print(f"{log_prefix} Chunk {i+1}/{len(chunks)} done. (~{avg:.1f} docs/s total)")

# ── Manager Logic (Launches subprocesses) ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Internal flag for worker process")
    parser.add_argument("--gpu_id", type=int, default=0, help="Assigned GPU ID")
    args = parser.parse_args()

    # Create output dir
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── BRANCH: WORKER PROCESS ────────────────────────────────────
    if args.worker:
        run_worker(args.gpu_id, NUM_GPUS)
        return

    # ── BRANCH: MANAGER PROCESS ───────────────────────────────────
    processes = []
    print(f"🚀 Launching {NUM_GPUS} independent workers...")
    
    for i in range(NUM_GPUS):
        # Create a Copy of Environment
        env = os.environ.copy()
        
        # STRICT ISOLATION:
        # Each worker sees only ONE GPU. It thinks it is GPU 0.
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        
        # Remove torchrun/distributed variables just to be safe
        for var in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            if var in env: del env[var]

        # Launch the command: python this_script.py --worker --gpu_id X
        cmd = [sys.executable, __file__, "--worker", "--gpu_id", str(i)]
        
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        print(f"   └── Launched Worker {i} (PID {p.pid}) on Physical GPU {i}")

    print("waiting for workers to finish...")
    exit_codes = [p.wait() for p in processes]
    print("All workers finished. Exit codes:", exit_codes)

if __name__ == "__main__":
    main()
