# ATPO Training - Running Notes

## Directory Layout

```
/scratch/user/debajoym98_tamu.edu/ATPO/
├── ATPO/                          # Main ATPO codebase
│   ├── scripts/
│   │   ├── ATPO_qwen3_4B.sh      # Training config & launch script
│   │   └── config/
│   │       └── ppo_trainer_dr.yaml  # Hydra YAML config
│   └── verl_atpo/                 # verl library (custom fork)
│       └── verl/
│           ├── trainer/
│           │   ├── main_ppo.py    # Entry point (Ray init + Hydra)
│           │   └── ppo/
│           │       └── ray_trainer.py  # RayPPOTrainer
│           └── workers/rollout/tools/
│               └── search_tool.py # BingSearchTool (patched for local RAG)
├── models/Qwen3-4B/              # Pre-downloaded model weights
├── rl_datasets/hotpotqa/         # train.parquet, test.parquet
├── rag_data/                     # FAISS index + wiki corpus for RAG server
│   ├── e5_Flat.index
│   └── wiki-18.jsonl
├── rag_server/
│   └── retrieval_server.py       # FastAPI RAG server (e5-base-v2)
├── run_atpo.sh                   # SLURM submission script (you run this)
├── logs/                         # Training outputs, checkpoints, W&B logs
└── .gitignore
```

## Conda Environments

Two separate conda environments are required:

| Environment | Path | Purpose |
|---|---|---|
| `atpo` | `/scratch/user/debajoym98_tamu.edu/conda_envs/atpo` | Main training (PyTorch, vLLM, Ray, transformers) |
| `retriever` | Standard conda path (`retriever`) | RAG server only (faiss, sentence-transformers) |

The `atpo` env lives under `/scratch` (not `~/.conda/envs`) to avoid NFS quota issues.

## How to Submit a Job

```bash
cd /scratch/user/debajoym98_tamu.edu/ATPO
sbatch run_atpo.sh
```

This submits `run_atpo.sh` to SLURM which:
1. Activates the `atpo` conda env
2. Cleans up stale Ray processes
3. Starts the RAG retrieval server (in the `retriever` env) on `http://127.0.0.1:8000`
4. Waits up to 5 minutes for the RAG server to be healthy
5. Runs `ATPO/scripts/ATPO_qwen3_4B.sh` which launches the PPO training via Hydra + Ray

## How to Monitor

### Check job queue
```bash
squeue -u $USER
```

### Watch SLURM output in real time
```bash
# Find the job ID from squeue, then:
tail -f slurm-atpo_qwen3_4B-<JOBID>.out
```

### Check Weights & Biases
The run logs to W&B project `ATPO` under experiment `multihop_qwen3_4B`.
Look for the W&B URL printed in the SLURM output:
```
wandb: View run at https://wandb.ai/...
```

### Check RAG server status
The RAG server log is written to:
```
/scratch/user/debajoym98_tamu.edu/ATPO/rag_server_<JOBID>.log
```

## SLURM Resource Configuration (run_atpo.sh)

| Setting | Value | Notes |
|---|---|---|
| Partition | `short` | 4-hour max wall time |
| Nodes | 1 | Single-node training |
| GPUs | 8 | Full DGX node |
| CPUs | 48 | |
| Memory | 200G | |
| Time limit | 4:00:00 | Increase if needed (change partition too) |
| `--exclusive` | yes | Prevents sharing the node |

To change wall time, edit both `#SBATCH -t` and `#SBATCH --partition` in `run_atpo.sh`.

## Key Files Modified (vs. original ATPO repo)

### 1. `search_tool.py` — Local RAG instead of Brightdata API

**File:** `ATPO/verl_atpo/verl/workers/rollout/tools/search_tool.py`

The `execute()` method was changed to POST to the local RAG server instead of the external Brightdata API:

```python
# Original: called https://api.brightdata.com/request
# Now: calls local RAG server
payload = {"queries": [query], "topk": 3, "return_scores": True}
response = requests.post("http://127.0.0.1:8000/retrieve", json=payload)
```

Cache read/write is commented out since the local server is fast enough.

### 2. `main_ppo.py` — Ray init fixes

**File:** `ATPO/verl_atpo/verl/trainer/main_ppo.py`

Three changes to `run_ppo()`:
- `include_dashboard=False` — avoids dashboard startup issues
- `_temp_dir` passed from `RAY_TMPDIR` env var — keeps Unix socket paths short (< 107 bytes)
- `num_cpus` read from config (set to `null` to let Ray auto-detect)

### 3. `ATPO_qwen3_4B.sh` — Paths, NCCL, and config fixes

**File:** `ATPO/scripts/ATPO_qwen3_4B.sh`

Changes:
- **Data paths**: Point to `/scratch/.../rl_datasets/hotpotqa/{train,test}.parquet`
- **Model path**: `/scratch/.../models/Qwen3-4B`
- **NCCL settings**: Stripped cluster-specific InfiniBand/bond settings. Set `NCCL_IB_DISABLE=1` (single-node, GPUs use NVLink)
- **Feature flags**: Defined `ENABLE_ENTROPY_BALANCED_CLIPPING`, `ENABLE_ENTROPY_BALANCED_ADVANTAGE`, `ENABLE_DYNAMIC_ROLLOUTS`, `ENABLE_MULTI_TURN`
- **tool_config_path**: Set to `verl_atpo/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml` (required when `multi_turn.enable=True`)
- **Search config**: `API_KEY="unused_local_rag_server"` (constructor requires it but local RAG doesn't use it)
- **ray_init.num_cpus=null**: Lets Ray auto-detect instead of conflicting with existing cluster

### 4. `run_atpo.sh` — SLURM wrapper script (created)

Handles: conda activation, Ray cleanup, RAG server lifecycle, cleanup trap.

### 5. Ray dashboard agent patch

**File:** `/scratch/user/debajoym98_tamu.edu/conda_envs/atpo/lib/python3.10/site-packages/ray/dashboard/modules/aggregator/multi_consumer_event_buffer.py`

Line 47 changed from:
```python
self._has_new_events_to_consume = asyncio.Condition(self._lock)
```
to:
```python
self._has_new_events_to_consume = asyncio.Condition()
```

This fixes a `ValueError: loop argument must agree with lock` bug in Ray 2.54.0 with Python 3.10. Without this patch, the dashboard agent crashes and Ray fails to start.

### 6. `.gitignore`

Excludes `models/`, `rag_data/`, `rl_datasets/` from git (they total ~142 GB).

## Package Versions (atpo conda env)

**Environment location:** `/scratch/user/debajoym98_tamu.edu/conda_envs/atpo`  
**Python:** 3.10.0  
**CUDA:** 12.4

### Core ML packages

| Package | Version | Install command |
|---|---|---|
| torch | 2.6.0+cu124 | `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| torchvision | 0.21.0+cu124 | `pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` |
| vllm | 0.8.4 | `pip install vllm==0.8.4` |
| transformers | 4.51.3 | `pip install transformers==4.51.3` |
| flash-attn | 2.7.4.post1 | `pip install flash-attn --no-build-isolation` |
| deepspeed | 0.18.8 | `pip install deepspeed==0.18.8` |
| accelerate | 1.13.0 | `pip install accelerate==1.13.0` |
| peft | 0.18.1 | `pip install peft==0.18.1` |

### Distributed / orchestration

| Package | Version | Install command |
|---|---|---|
| ray | 2.54.0 | `pip install ray==2.54.0` |
| opentelemetry-sdk | 1.40.0 | `pip install "opentelemetry-sdk>=1.40.0"` |
| opentelemetry-api | 1.40.0 | `pip install "opentelemetry-api>=1.40.0"` |
| opentelemetry-semantic-conventions | 0.61b0 | `pip install "opentelemetry-semantic-conventions>=0.50b0"` |
| grpcio | 1.71.2 | `pip install grpcio` |

### Config / logging / data

| Package | Version | Install command |
|---|---|---|
| hydra-core | 1.3.2 | `pip install hydra-core==1.3.2` |
| omegaconf | 2.3.0 | `pip install omegaconf==2.3.0` |
| wandb | 0.25.1 | `pip install wandb` |
| datasets | 4.8.4 | `pip install datasets` |
| pandas | 2.3.3 | `pip install pandas` |
| numpy | 2.2.6 | `pip install numpy` |

### Tokenization / serialization

| Package | Version | Install command |
|---|---|---|
| tokenizers | 0.21.4 | `pip install tokenizers==0.21.4` |
| sentencepiece | 0.2.1 | `pip install sentencepiece` |
| safetensors | 0.7.0 | `pip install safetensors` |
| protobuf | 4.25.8 | `pip install protobuf==4.25.8` |

### Utilities

| Package | Version | Install command |
|---|---|---|
| setuptools | 77.0.3 | `pip install "setuptools<78"` |
| requests | 2.32.5 | `pip install requests` |
| scipy | 1.15.3 | `pip install scipy` |
| langid | 1.1.6 | `pip install langid` |
| fastapi | 0.135.2 | `pip install fastapi` |
| uvicorn | 0.42.0 | `pip install uvicorn` |

### Quick full install (in order)

```bash
conda create -n atpo python=3.10 -y
conda activate atpo

# PyTorch + CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Flash Attention (requires torch first)
pip install flash-attn --no-build-isolation

# vLLM (may pull in its own deps)
pip install vllm==0.8.4

# Pin transformers (vllm may install a newer incompatible version)
pip install transformers==4.51.3

# Ray + opentelemetry (vllm may downgrade opentelemetry, so install after)
pip install ray==2.54.0
pip install "opentelemetry-sdk>=1.40.0" "opentelemetry-api>=1.40.0" "opentelemetry-semantic-conventions>=0.50b0"

# Training / config
pip install deepspeed==0.18.8 accelerate==1.13.0 peft==0.18.1
pip install hydra-core==1.3.2 omegaconf==2.3.0
pip install wandb datasets pandas scipy

# Tokenization / misc
pip install sentencepiece safetensors protobuf==4.25.8 langid
pip install "setuptools<78"  # needed for pkg_resources

# RAG server deps (if not using separate retriever env)
pip install fastapi uvicorn
```

### Critical version constraints

- **torchvision must be 0.21.x** for torch 2.6; v0.22 is incompatible
- **transformers must be 4.51.x**; v4.57+ breaks vllm 0.8.4 (`undefined symbol` in `_C.abi3.so`)
- **opentelemetry-sdk must be >= 1.40.0**; Ray 2.54.0 needs it but vllm pins lower
- **setuptools must be < 78**; provides `pkg_resources` needed by verl
- **ray 2.54.0** needs the dashboard agent patch (see above) for Python 3.10

If you ever reinstall vllm, it may downgrade opentelemetry and transformers. After reinstalling, run:
```bash
pip install "opentelemetry-sdk>=1.40.0" "opentelemetry-api>=1.40.0" "transformers==4.51.3"
```

## Common Errors and Fixes

### "tool_config_path must be set when enabling multi_turn with tool"
`ATPO_qwen3_4B.sh` must pass `actor_rollout_ref.rollout.multi_turn.tool_config_path=<path>` when `multi_turn.enable=True`.

### "Bootstrap : no socket interface found" (NCCL)
The original script hardcodes `NCCL_SOCKET_IFNAME=bond1` which doesn't exist on DGX nodes. Fix: remove it and set `NCCL_IB_DISABLE=1` for single-node runs.

### "AF_UNIX path length cannot exceed 107 bytes"
Ray's Unix sockets fail if the temp dir path is too long. Fix: use `/tmp/ray_${SLURM_JOB_ID}` (short path), not `/scratch/...`.

### "ValueError: loop argument must agree with lock"
Ray 2.54.0 + Python 3.10 bug. Fix: patch `multi_consumer_event_buffer.py` (see above).

### "When connecting to an existing cluster, num_cpus must not be provided"
Stale Ray processes on the node. Fix: `ray stop --force` + `pkill` before starting, and set `ray_init.num_cpus=null`.

### "vllm/_C.abi3.so: undefined symbol"
vllm/torch/transformers version mismatch. Fix: pin `transformers==4.51.3` and `torchvision==0.21.0+cu124`.

### "No module named 'pkg_resources'"
Missing `setuptools`. Fix: `pip install "setuptools<78"`.

## W&B Login

W&B credentials are stored in `~/.netrc` after logging in once. To log in:
```bash
conda activate /scratch/user/debajoym98_tamu.edu/conda_envs/atpo
wandb login
```
The SLURM job picks up the saved credentials automatically.

## Cleanup

To clean up after failed runs:
```bash
# Remove failed SLURM logs
rm slurm-atpo_qwen3_4B-<FAILED_JOBID>.{out,err}

# Remove failed training log directories
rm -rf logs/multihop_qwen3_4B/<TIMESTAMP_OF_FAILED_RUN>/

# Remove RAG server logs
rm rag_server_<JOBID>.log
```
