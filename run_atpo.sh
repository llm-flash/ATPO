#!/bin/bash
#SBATCH -J atpo_qwen3_4B
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH -t 4:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --mem=200G
#SBATCH --exclusive

source ~/.bashrc
conda activate /scratch/user/debajoym98_tamu.edu/conda_envs/atpo

# export WANDB_API_KEY="your_wandb_api_key_here"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "================"

nvidia-smi

# Clean up any stale Ray processes and start fresh
ray stop --force 2>/dev/null || true
pkill -u $USER -f "raylet|gcs_server|ray::" 2>/dev/null || true
sleep 3
rm -rf /tmp/ray/ /tmp/ray_* 2>/dev/null || true

# Ray needs short Unix socket paths (max 107 bytes)
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export RAY_GCS_BOOTSTRAPPING_TIMEOUT=120
export RAY_DASHBOARD_AGENT_ENABLED=0
echo "Ray temp dir: $RAY_TMPDIR"

# ===================== Launch RAG Retrieval Server =====================
RAG_DIR="/scratch/user/debajoym98_tamu.edu/ATPO"
RAG_DATA="${RAG_DIR}/rag_data"
RAG_LOG="${RAG_DIR}/rag_server_${SLURM_JOB_ID}.log"

echo "Starting RAG retrieval server (using 'retriever' conda env)..."
conda run -n retriever python3 ${RAG_DIR}/rag_server/retrieval_server.py \
    --index_path ${RAG_DATA}/e5_Flat.index \
    --corpus_path ${RAG_DATA}/wiki-18.jsonl \
    --topk 3 \
    --retriever_model intfloat/e5-base-v2 \
    > "${RAG_LOG}" 2>&1 &
RAG_PID=$!

cleanup() {
    echo "Cleaning up..."
    echo "=== Ray logs ==="
    echo "=== Dashboard agent log ===" && cat "$RAY_TMPDIR"/session_*/logs/dashboard_agent.log 2>/dev/null | tail -40
    echo "=== Runtime env agent log ===" && cat "$RAY_TMPDIR"/session_*/logs/runtime_env_agent.log 2>/dev/null | tail -20
    echo "=== Raylet err ===" && cat "$RAY_TMPDIR"/session_*/logs/raylet.err 2>/dev/null | tail -10
    kill $RAG_PID 2>/dev/null
    wait $RAG_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for RAG server to be ready (PID: $RAG_PID)..."
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:8000/docs > /dev/null 2>&1; then
        echo "RAG server is ready after ${ELAPSED}s"
        break
    fi
    if ! kill -0 $RAG_PID 2>/dev/null; then
        echo "ERROR: RAG server process died. Check ${RAG_LOG}"
        cat "${RAG_LOG}"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: RAG server failed to start within ${MAX_WAIT}s. Log:"
    cat "${RAG_LOG}"
    exit 1
fi

# ===================== Run ATPO Training =====================
echo "Starting ATPO training..."
bash /scratch/user/debajoym98_tamu.edu/ATPO/ATPO/scripts/ATPO_qwen3_4B.sh

echo "=== Job finished at $(date) ==="
