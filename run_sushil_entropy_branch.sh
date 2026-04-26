#!/bin/bash

#SBATCH --job-name=ATPO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --account=early-adopters
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_%j.log

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

# ==========================================
# Environment Setup
# ==========================================
ml CUDA/12.9.1
source ~/.bashrc
conda activate atpo_env

# ==========================================
# Ray Environment Preparation
# ==========================================
ray stop --force 2>/dev/null || true
pkill -u $USER -f "raylet|gcs_server|ray::" 2>/dev/null || true
rm -rf /tmp/ray/ /tmp/ray_* 2>/dev/null || true

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

unset RAY_ADDRESS
export RAY_GCS_BOOTSTRAPPING_TIMEOUT=120
export RAY_DASHBOARD_AGENT_ENABLED=0

# ==========================================
# Start Background RAG Server
# ==========================================
RAG_LOG="logs/rag_server_${SLURM_JOB_ID}.log"

conda run -n retriever_env \
    python rag_server/retrieval_server.py \
    --index_path rag_data/e5_Flat.index \
    --corpus_path rag_data/wiki-18.jsonl \
    --topk 3 \
    --retriever_model intfloat/e5-base-v2 \
    > "${RAG_LOG}" 2>&1 &
RAG_PID=$!

# ==========================================
# Cleanup Routine
# ==========================================
cleanup() {
    echo "Cleaning up processes..."
    kill $RAG_PID 2>/dev/null
    wait $RAG_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# ==========================================
# Health Check Wait Loop
# ==========================================
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

# ==========================================
# MAIN
# ==========================================
echo "Starting ATPO training..."
ENTROPY_METHOD=${1:?"Error: You must provide an entropy method (S1 or S2)."}
export ENTROPY_METHOD
bash ATPO/scripts/sushil_entropy_branch.sh

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."