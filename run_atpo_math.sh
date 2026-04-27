
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

cleanup() {
    echo "Cleaning up..."
    echo "=== Ray logs ==="
    echo "=== Dashboard agent log ===" && cat "$RAY_TMPDIR"/session_*/logs/dashboard_agent.log 2>/dev/null | tail -40
    echo "=== Runtime env agent log ===" && cat "$RAY_TMPDIR"/session_*/logs/runtime_env_agent.log 2>/dev/null | tail -20
    echo "=== Raylet err ===" && cat "$RAY_TMPDIR"/session_*/logs/raylet.err 2>/dev/null | tail -10
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# ===================== Run ATPO Training =====================
echo "Starting ATPO training..."
bash /data1/jcarleton/ATPO/ATPO/scripts/math_ATPO_qwen3_4B.sh

echo "=== Job finished at $(date) ==="
