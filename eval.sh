#!/bin/bash

#SBATCH --job-name=ATPO_Training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gpus=8
#SBATCH --time=4:00:00
#SBATCH --account=early-adopters
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_%j.log

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

# ==========================================
# Parse Arguments
# ==========================================
if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 <algorithm: atpo|cache> <dataset: hotpotqa|nq> <checkpoint_path>"
    exit 1
fi

ALGO_ARG=$(echo "$1" | tr '[:upper:]' '[:lower:]')
DATASET_ARG=$(echo "$2" | tr '[:upper:]' '[:lower:]')
CHECKPOINT_PATH=$(echo "$3" | sed 's:/*$::')

# ==========================================
# Resolve Relative Directories
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PARENT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"
echo "Project directory set to: $PROJECT_DIR"

PROJECT_NAME="CACHE_eval"

# 1. Algorithm specific settings
if [ "$ALGO_ARG" == "cache" ]; then
    ROLLOUT_MODE="sync_with_tool_tree_cache"
    EXPANSION_MODE="entropy"
    NODE_VALUE_MODE="child_mean"
    SEARCH_CACHE_PATH="${PROJECT_DIR}/search_cache/search_cache_entropy_branch.json"
    EXTRA_TOOL_ARGS="actor_rollout_ref.rollout.tools.call_limit=6"
elif [ "$ALGO_ARG" == "atpo" ]; then
    ROLLOUT_MODE="sync_with_tool_tree"
    EXPANSION_MODE="entropy"
    NODE_VALUE_MODE="child_softmax"
    SEARCH_CACHE_PATH="${PROJECT_DIR}/search_cache/search_cache.json"
    EXTRA_TOOL_ARGS="actor_rollout_ref.rollout.tools.call_limit=6"
else
    echo "Error: Invalid algorithm '$ALGO_ARG'. Must be 'atpo' or 'cache'."
    exit 1
fi

# 2. Dataset specific settings
# Due to differences in the columns used for these datasets, I cannot include hotpotqa and the others all at once
if [ "$DATASET_ARG" == "hotpotqa" ]; then
    TRAIN_FILES="${PROJECT_DIR}/rl_datasets/hotpotqa/train.parquet"
    VALID_FILES="[\"${PROJECT_DIR}/rl_datasets/hotpotqa_test.parquet\"]"
    # VALID_FILES="[\"${PROJECT_DIR}/rl_datasets/2wikimultihopqa_test.parquet\",\"${PROJECT_DIR}/rl_datasets/bamboogle_test.parquet\",\"${PROJECT_DIR}/rl_datasets/musique_test.parquet\"]"
elif [ "$DATASET_ARG" == "other_multihop" ]; then
    TRAIN_FILES="${PROJECT_DIR}/rl_datasets/hotpotqa/train.parquet"
    VALID_FILES="[\"${PROJECT_DIR}/rl_datasets/2wikimultihopqa_test.parquet\",\"${PROJECT_DIR}/rl_datasets/bamboogle_test.parquet\",\"${PROJECT_DIR}/rl_datasets/musique_test.parquet\"]"
elif [ "$DATASET_ARG" == "nq" ]; then
    TRAIN_FILES="${PROJECT_DIR}/rl_datasets/nq/train.parquet"
    VALID_FILES="[\"${PROJECT_DIR}/rl_datasets/nq_test.parquet\",\"${PROJECT_DIR}/rl_datasets/popqa_test.parquet\",\"${PROJECT_DIR}/rl_datasets/triviaqa_test.parquet\"]"
else
    echo "Error: Invalid dataset '$DATASET_ARG'. Must be 'hotpotqa' or 'nq'."
    exit 1
fi

ACTOR_MODEL_PATH="${CHECKPOINT_PATH}"
RESUME_MODE="resume_path"

EXPERIMENT_NAME="${ALGO_ARG}_${DATASET_ARG}_EVAL"

# ==========================================
# Environment Setup
# ==========================================
ml CUDA/12.9.1
source ~/.bashrc
conda activate atpo_env

export NCCL_DEBUG="WARN"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPU_PER_NODE=8
export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
export VERL_LOGGING_LEVEL=WARN
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export RAY_DEBUG=1
export PYTHONPATH=${PARENT_DIR}/verl_atpo:$PYTHONPATH

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

cleanup() {
    echo "Cleaning up processes..."
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

# ==========================================
# Output Paths & WandB Preparation
# ==========================================
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="${PROJECT_DIR}/logs/${EXPERIMENT_NAME}/${CURRENT_DATE}/"

if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

mkdir -p "$SAVE_PATH/rollout"
mkdir -p "$SAVE_PATH/validation"

# ==========================================
# MAIN TRAINING (VERL)
# ==========================================
echo "Starting $PROJECT_NAME training ($EXPERIMENT_NAME)..."

python3 -m verl.trainer.main_ppo \
    --config-path="${PARENT_DIR}/scripts/config" \
    --config-name="ppo_trainer_dr.yaml" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key="prompt" \
    data.train_batch_size=8 \
    data.val_batch_size=256 \
    data.max_prompt_length=2000 \
    data.max_response_length=6192 \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss=gspo_turn \
    actor_rollout_ref.actor.clip_ratio_low=3e-3 \
    actor_rollout_ref.actor.clip_ratio_high=4e-3 \
    actor_rollout_ref.actor.enable_entropy_balanced_clipping=False \
    actor_rollout_ref.actor.enable_entropy_balanced_advantage=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(2000+6192))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enable_dynamic_rollouts=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(2000+6192))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=22 \
    actor_rollout_ref.rollout.initial_rollouts=10 \
    actor_rollout_ref.rollout.expansion_mode=${EXPANSION_MODE} \
    actor_rollout_ref.rollout.expansion_iterations=2 \
    actor_rollout_ref.rollout.beam_size=6 \
    actor_rollout_ref.rollout.samples_per_tree=22 \
    actor_rollout_ref.rollout.branch_probability=0.5 \
    actor_rollout_ref.rollout.entropy_weight=0.2 \
    actor_rollout_ref.rollout.leaf_value_norm=True \
    actor_rollout_ref.rollout.node_value_mode=${NODE_VALUE_MODE} \
    actor_rollout_ref.rollout.node_adv_mode=node_value \
    ${EXTRA_TOOL_ARGS} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key="unused_local_rag_server" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${PARENT_DIR}/verl_atpo/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(2000+6192))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="naive" \
    custom_reward_function.path="${PARENT_DIR}/verl_atpo/verl/utils/reward_score/deep_research_em.py" \
    custom_reward_function.name="compute_score" \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU_PER_NODE} \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    +trainer.val_only=True \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=True \
    trainer.rollout_data_dir="${SAVE_PATH}/rollout" \
    trainer.validation_data_dir="${SAVE_PATH}/validation" \
    ray_init.num_cpus=null \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."