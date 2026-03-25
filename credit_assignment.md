# ATPO Credit Assignment Pipeline — Detailed Notes

## Overview

ATPO (Agentic Tree Policy Optimization) performs credit assignment by building a **tree of multi-turn reasoning trajectories**, scoring leaf nodes with outcome rewards, propagating values **bottom-up through the tree** using entropy-weighted aggregation, computing per-node advantages, and mapping those advantages to **token-level signals** for policy gradient updates. The policy loss used is **GSPO-turn** (sequence-level importance ratios computed per-turn).

This document traces the full pipeline from tree construction through to gradient computation, identifies each relevant code section, and explains how to modify each stage.

---

## Pipeline Stages at a Glance

```
1. Tree Construction & Entropy-Based Expansion
        ↓
2. Leaf Reward Scoring (Exact Match)
        ↓
3. Leaf Value Normalization (per-tree z-score)
        ↓
4. Bottom-Up Value Propagation (entropy-weighted softmax)
        ↓
5. Node Advantage Computation (node_value / diff_parent / diff_global / …)
        ↓
6. Token-Level Advantage Mapping (root→leaf path, incremental tokens)
        ↓
7. Policy Gradient Loss (GSPO-turn with per-turn importance ratios)
```

---

## 1. Tree Construction & Entropy-Based Expansion

### File
`verl_atpo/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools_tree_offline.py`

### How It Works

Each input prompt gets **one tree** with a single root node. The root spawns `initial_rollouts` child branches (default: 10). Each child is generated to completion (tool calls → tool results → continue until EOS or max length). After the initial phase, the tree is expanded for `expansion_iterations` rounds (default: 2).

#### Entropy Calculation (line 913-914)
After each generation step, the model's token-level log-probabilities are collected and entropy is computed as the **negative average log-prob**:

```python
node.entropy = -sum(logprobs) / len(logprobs)
```

Higher entropy means the model was less certain — these nodes are prioritized for branching.

#### Node Selection for Expansion (lines 341-403, method `sample_expansion_nodes`)
When `expansion_mode='entropy'` (the current config), candidate non-leaf nodes are scored by:

```python
prob = entropy_now - 0.05 * num_existing_branches
```

Where `num_existing_branches` is the number of sibling branches at the parent. Nodes are sorted by this score (descending) and the top `beam_size` (default: 6) are selected for expansion.

#### Branch Creation (lines 1409-1450)
For each selected node, a **sibling branch** is created from its parent. The new branch copies the selected node's token state but **rolls back the last generation** (`last_rollout_length` tokens removed). This lets the model re-generate from the same tool-result boundary with different sampling.

### Current Config Values (from `ATPO_qwen3_4B.sh`)
| Parameter | Value | Description |
|---|---|---|
| `initial_rollouts` | 10 | Initial branches from root |
| `expansion_mode` | `entropy` | Use entropy for node selection |
| `expansion_iterations` | 2 | Number of expansion rounds |
| `beam_size` | 6 | Nodes selected per expansion |
| `samples_per_tree` | 22 | Final leaves sampled from tree |
| `branch_probability` | 0.5 | (unused in offline mode) |
| `entropy_weight` | 0.2 | (unused in current entropy mode) |

### How to Change

- **Switch to random expansion**: Set `EXPANSION_MODE="random"` in `ATPO_qwen3_4B.sh`
- **Increase branching diversity**: Increase `EXPANSION_ITERATIONS` or `BEAM_SIZE`
- **Reduce tree size (faster training)**: Decrease `INITIAL_ROLLOUTS` and `SAMPLES_PER_TREE` (must keep `ROLLOUT_N == SAMPLES_PER_TREE`)
- **Change entropy scoring formula**: Modify `sample_expansion_nodes()` at line 375. Currently uses raw entropy; you could use `entropy_delta = entropy_now - entropy_init` or add randomness back:
  ```python
  prob = random.random() + entropy_weight * entropy_delta  # was commented out
  ```
- **Change branch penalty**: Modify the `0.05` coefficient at line 386

---

## 2. Leaf Reward Scoring

### File
`verl_atpo/verl/utils/reward_score/deep_research_em.py` — function `compute_score` (line 306)

### How It Works

After all leaves are generated, rewards are computed via `compute_reward()` which calls `compute_score()`. The scoring logic:

1. **Format validation**: Checks for `<think>`, `<answer>`, `\boxed{}` tags
2. **Answer extraction**: Extracts answer from `<answer>\boxed{...}</answer>`
3. **F1 scoring**: Token-level F1 between predicted and ground truth
4. **EM scoring**: Exact match after normalization (lowercase, remove articles/punctuation)
5. **Final reward**:
   - `f1 > 0` AND both `</search>` and `</python>` used → `score = f1 + 0.1` (multi-tool bonus)
   - `f1 > 0` (single tool) → `score = em_score` (0 or 1)
   - `f1 == 0` → `score = 0` (wrong but valid format)
   - Bad format → `score = -1`

### Leaf Value Assignment (lines 1660-1673 of tree_offline.py)
For each leaf, the reward at the **last valid response token position** is extracted:

```python
leaf_value = reward_tensor[i, valid_response_length - 1].item()
leaf_node.value = leaf_value
```

### How to Change

- **Use F1 instead of EM**: In `compute_score()`, change line 377 from `result["score"] = em_score` to `result["score"] = f1_score`
- **Add format reward shaping**: Give partial credit for correct format even with wrong answer (currently gives 0)
- **Add length penalty**: Penalize very long responses by modifying the score

---

## 3. Leaf Value Normalization

### File
`vllm_rollout_with_tools_tree_offline.py`, lines 1720-1745

### How It Works (when `leaf_value_norm=True`)

Per-tree z-score normalization:
```python
leaf.value = (leaf.value - mean_value) / (std_value + epsilon)
```

Where `mean_value` and `std_value` are computed over all leaves **within the same tree**. This ensures that within each tree, at least some leaves have positive advantage and some have negative, enabling contrastive learning regardless of whether the question is easy or hard.

### Current Config
`actor_rollout_ref.rollout.leaf_value_norm=True` (set in `ATPO_qwen3_4B.sh` line 174)

### How to Change

- **Disable normalization**: Set `leaf_value_norm=False`. Raw rewards (0/1/−1) will be used directly.
- **Use different normalization**: Modify lines 1736-1743. For example, min-max normalization:
  ```python
  min_val = np.min(leaf_values_array)
  max_val = np.max(leaf_values_array)
  leaf.value = (leaf.value - min_val) / (max_val - min_val + epsilon)
  ```

---

## 4. Bottom-Up Value Propagation

### File
`vllm_rollout_with_tools_tree_offline.py`, method `compute_value_from_children` (lines 559-608)

### How It Works

Three modes available, controlled by `node_value_mode`:

#### `child_softmax` (current default)
Non-leaf node value is the **entropy-weighted sum** of its children:
```python
exp_entropies = np.exp(child_entropies_array)
softmax_weights = exp_entropies / np.sum(exp_entropies)
self.value = np.sum(softmax_weights * child_values_array)
```

Higher-entropy children (where the model was more uncertain) get **more weight** in the parent's value. The intuition: uncertain branches reveal more about what the model needs to learn.

#### `child_mean`
Simple average of all child values:
```python
self.value = np.mean(child_values)
```

#### `leaf_mean` (lines 1753-1767)
Each non-leaf node's value is the mean of **all descendant leaf values** (not just direct children):
```python
leaf_values = [leaf.value for leaf in node.get_all_leaves() if leaf.value is not None]
node.value = np.mean(leaf_values)
```

### Invocation (line 1751)
```python
root.compute_value_from_children(mode=self.node_value_mode)
```

### Current Config
`actor_rollout_ref.rollout.node_value_mode=child_softmax`

### How to Change

- **Switch to mean aggregation**: Set `node_value_mode=child_mean` in the hydra override
- **Switch to leaf mean**: Set `node_value_mode=leaf_mean`
- **Custom aggregation**: Add a new mode in `compute_value_from_children()`. For example, max-child:
  ```python
  elif mode == 'child_max':
      self.value = np.max(child_values)
  ```
  Then set `node_value_mode=child_max` in the script.

---

## 5. Node Advantage Computation

### File
`vllm_rollout_with_tools_tree_offline.py`, lines 1772-1814

### How It Works

Five modes available, controlled by `node_adv_mode`:

#### `node_value` (current default)
Directly uses the propagated value as advantage:
```python
node.advantage = node.value
```

#### `diff_parent`
Advantage = node value minus parent value:
```python
node.advantage = node.value - node.parent_node.value
```
Captures the **marginal improvement** from parent to child.

#### `diff_global`
Advantage = node value minus root value:
```python
node.advantage = node.value - root.value
```

#### `diff_localglobal`
Combines both local and global:
```python
node.advantage = (node.value - root.value) + (node.value - node.parent_node.value)
```

#### `vanilla` (lines 1680-1713)
Bypasses tree-based credit assignment entirely. Uses standard **GRPO outcome-level advantage** (identical to vanilla verl):
- `token_level_scores = reward_tensor`
- Optionally applies KL penalty
- Calls `compute_advantage()` with the GRPO estimator

### Current Config
`actor_rollout_ref.rollout.node_adv_mode=node_value`

### How to Change

- **Try parent-relative advantage**: Set `node_adv_mode=diff_parent`. This makes the training signal more local: each node's advantage depends only on how much it improved over its parent.
- **Ablation with vanilla GRPO**: Set `node_adv_mode=vanilla` to disable tree-based process rewards entirely
- **Add new mode**: Add an `elif` block at line 1802+. For example, a depth-weighted mode:
  ```python
  elif self.node_adv_mode == 'depth_weighted':
      for root in root_nodes:
          all_nodes = [root] + root.get_subtree_nodes()
          max_depth = max(n.depth for n in all_nodes) or 1
          for node in all_nodes:
              depth_weight = node.depth / max_depth
              node.advantage = node.value * depth_weight
  ```
  Then pass `node_adv_mode=depth_weighted` in the hydra command.

---

## 6. Token-Level Advantage Mapping

### File
`vllm_rollout_with_tools_tree_offline.py`, lines 1827-1856

### How It Works

For each sampled leaf, the algorithm traces the **root → leaf path** and assigns each node's advantage to its **incremental tokens** (the tokens that node added beyond its parent):

```python
for j in range(1, len(path_nodes)):
    node = path_nodes[j]
    parent_node = path_nodes[j - 1]

    parent_len = len(parent_node.curr_token_ids) - len(parent_node.prompt_token_ids)
    node_len   = len(node.curr_token_ids) - len(node.prompt_token_ids)

    if node_len > parent_len:
        start_idx = parent_len
        end_idx   = min(node_len, response_length)
        token_level_advantages[i, start_idx:end_idx] = node.advantage
```

This means:
- Root node's direct generation → gets root's advantage
- After first tool call, the new node's generation → gets that node's advantage
- And so on for each depth in the tree

**Tool result tokens** (inserted by the system) have `result_mask=0` and are excluded from loss via `loss_mask`.

### The `loss_mask` (line 1614)
```python
loss_mask = loss_mask * response_attention_mask
```
Ensures only model-generated tokens (not tool results or padding) participate in the loss.

### How to Change

- **Uniform advantage per leaf**: Replace the loop with a flat assignment:
  ```python
  token_level_advantages[i, :] = leaf_node.value
  ```
  This makes every token in a trajectory get the same advantage (like standard GRPO).

- **Discount by depth**: Weight advantages by how deep in the tree they occur:
  ```python
  depth_discount = 0.9 ** node.depth
  token_level_advantages[i, start_idx:end_idx] = node.advantage * depth_discount
  ```

---

## 7. Policy Gradient Loss — GSPO-turn

### File
`verl_atpo/verl/trainer/ppo/core_algos.py`, function `compute_policy_loss_gspo_turn` (lines 970-1092)

### Dispatched from
`verl_atpo/verl/workers/actor/dp_actor.py`, lines 558-568

### How It Works

GSPO-turn is a variant of GSPO that computes **per-turn importance ratios** instead of per-sequence ratios. This is critical for multi-turn scenarios where the response mask has multiple segments (model turns interspersed with tool results).

#### Step-by-step:

1. **Identify turns** from `response_mask` (lines 1008-1014):
   ```python
   mask_padded = cat([0, mask, 0])
   diff = mask_padded[1:] - mask_padded[:-1]
   turn_starts = where(diff == 1)
   turn_ends   = where(diff == -1)
   ```

2. **Per-turn KL** (lines 1025-1029):
   For each turn segment `[start:end]`:
   ```python
   turn_negative_approx_kl_mean = negative_approx_kl[start:end].sum() / turn_length
   turn_importance_ratio_per_token[batch_idx, start:end] = turn_negative_approx_kl_mean
   ```

3. **Turn-level importance ratio** (lines 1059-1063):
   ```python
   log_ratio = log_prob - log_prob.detach() + turn_importance_ratio_per_token.detach()
   turn_importance_ratio = exp(clamp(log_ratio, max=10.0))
   ```
   The `log_prob - log_prob.detach()` trick ensures gradients flow through the current policy while the turn-level ratio acts as a stop-gradient scaling factor.

4. **Clipped loss** (lines 1065-1070):
   ```python
   pg_losses1 = -advantages * turn_importance_ratio
   pg_losses2 = -advantages * clamp(turn_importance_ratio, 1 - clip_low, 1 + clip_high)
   pg_losses  = max(pg_losses1, pg_losses2)
   pg_loss    = agg_loss(pg_losses, response_mask, "seq-mean-token-mean")
   ```

5. **Turn-KL entropy metric** (lines 1040-1054):
   For each sample, computes the Shannon entropy of the turn-level KL distribution (normalized by log(num_turns)). This measures how evenly the policy update is distributed across turns.

### Current Config
| Parameter | Value | Location |
|---|---|---|
| `policy_loss` | `gspo_turn` | `ATPO_qwen3_4B.sh` line 146 |
| `clip_ratio_low` | `3e-3` | line 147 |
| `clip_ratio_high` | `4e-3` | line 148 |
| `loss_agg_mode` | hardcoded `seq-mean-token-mean` | core_algos.py line 1070 |

### Available Policy Loss Alternatives (dp_actor.py lines 547-616)
| Value | Function | Description |
|---|---|---|
| `gspo` | `compute_policy_loss_gspo` | Sequence-level importance ratio |
| `gspo_turn` | `compute_policy_loss_gspo_turn` | **Current** — turn-level importance ratio |
| `gppo` | `compute_policy_loss_gppo` | Group PPO with soft clip |
| `gppo_half` | `compute_policy_loss_gppo_half` | GPPO with half-soft clip |
| `grpo` | `compute_policy_loss_entropy_balanced_clipping` | GRPO with entropy-balanced clipping |

### How to Change

- **Switch to sequence-level GSPO**: Set `policy_loss=gspo`
- **Switch to standard GRPO**: Set `policy_loss=grpo`
- **Adjust clipping range**: Modify `clip_ratio_low` and `clip_ratio_high` in the hydra override
- **Add a new loss**: Define a new function in `core_algos.py`, import it in `dp_actor.py`, and add an `elif` branch

---

## Complete Data Flow Diagram

```
Input Prompt
    │
    ▼
Root Node (1 per sample)
    │
    ├── Phase 1: Build initial_rollouts children (10 branches)
    │
    ├── Phase 2: Generate complete chains for each branch
    │            (model generates → tool call → tool result → continue → EOS)
    │            Entropy computed at each generation step
    │
    ├── Phase 3: Expansion (2 iterations)
    │   ├── Select beam_size=6 high-entropy non-leaf nodes
    │   ├── Create sibling branches (rollback last generation)
    │   └── Generate complete chains for new branches
    │
    ├── Phase 4: Sample samples_per_tree=22 leaves from tree
    │            (prune excess, duplicate if too few)
    │
    ├── Phase 5: Pad/stack outputs into tensors
    │
    └── Phase 6: Credit Assignment
        │
        ├── 6a. Compute reward for each leaf (compute_score → EM/F1)
        │
        ├── 6b. Normalize leaf values per-tree: z-score
        │        leaf.value = (value - mean) / (std + eps)
        │
        ├── 6c. Propagate values bottom-up: child_softmax
        │        parent.value = Σ softmax(child_entropy) * child.value
        │
        ├── 6d. Compute node advantages: node_value
        │        node.advantage = node.value
        │
        └── 6e. Map to token-level advantages
                 For each leaf, trace root→leaf path
                 Each node's advantage → its incremental tokens
                     │
                     ▼
            token_level_advantages tensor (batch, response_length)
                     │
                     ▼
            GSPO-turn policy loss (per-turn importance ratios)
```

---

## Key Config Knobs Summary

All set via hydra overrides in `ATPO_qwen3_4B.sh`:

| Knob | Config Key | Current Value | Alternatives |
|---|---|---|---|
| Tree expansion strategy | `rollout.expansion_mode` | `entropy` | `random` |
| Value propagation | `rollout.node_value_mode` | `child_softmax` | `child_mean`, `leaf_mean` |
| Advantage mode | `rollout.node_adv_mode` | `node_value` | `vanilla`, `diff_parent`, `diff_global`, `diff_localglobal` |
| Leaf normalization | `rollout.leaf_value_norm` | `True` | `False` |
| Policy loss | `actor.policy_loss` | `gspo_turn` | `gspo`, `gppo`, `gppo_half`, `grpo` |
| Clip range (low) | `actor.clip_ratio_low` | `3e-3` | Any float |
| Clip range (high) | `actor.clip_ratio_high` | `4e-3` | Any float |
| Advantage estimator | `algorithm.adv_estimator` | `grpo` | `gae` (needs critic) |

---

## File Reference Index

| File | Key Functions / Lines | Role |
|---|---|---|
| `verl_atpo/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools_tree_offline.py` | Full file (1882 lines) | Tree construction, generation, reward, credit assignment |
| — | `ToolTreeNode` (L224-608) | Tree node data structure |
| — | `sample_expansion_nodes` (L341-406) | Entropy-based node selection |
| — | `compute_value_from_children` (L559-608) | Bottom-up value propagation |
| — | `generate_sequences` (L1249-1881) | Main entry point (all 6 phases) |
| — | L913-914 | Entropy computation |
| — | L1660-1673 | Leaf value assignment from reward |
| — | L1720-1745 | Leaf value normalization |
| — | L1748-1770 | Bottom-up propagation dispatch |
| — | L1773-1814 | Node advantage computation |
| — | L1827-1856 | Token-level advantage mapping |
| `verl_atpo/verl/trainer/ppo/core_algos.py` | Full file (1466 lines) | All policy loss functions |
| — | `compute_grpo_outcome_advantage` (L119-174) | GRPO advantage (vanilla mode) |
| — | `compute_policy_loss_gspo_turn` (L970-1092) | Current policy loss |
| — | `compute_policy_loss_gspo` (L843-967) | Sequence-level GSPO |
| — | `agg_loss` (L457-492) | Loss aggregation modes |
| `verl_atpo/verl/utils/reward_score/deep_research_em.py` | `compute_score` (L306-384) | Outcome reward (EM/F1) |
| `verl_atpo/verl/workers/actor/dp_actor.py` | L547-616 | Policy loss dispatch |
| `verl_atpo/verl/trainer/ppo/ray_trainer.py` | `fit()` (L939-1246) | Training loop orchestration |
| `scripts/ATPO_qwen3_4B.sh` | Full file (201 lines) | All hydra config overrides |
| `scripts/config/ppo_trainer_dr.yaml` | (referenced by script) | Base Hydra config |

---

## Configuration: Entropy Branching Only (No Fancy Credit Assignment)

If you want to keep **entropy-based tree expansion** but use **standard GRPO outcome-level advantages** (no tree value propagation, no softmax weighting, no per-node advantages), change **one line** in `ATPO_qwen3_4B.sh`:

```
actor_rollout_ref.rollout.node_adv_mode=vanilla
```

### What this does

When `node_adv_mode=vanilla`, the code at line 1680 of `vllm_rollout_with_tools_tree_offline.py` takes a completely different path:

- **Skips** leaf value normalization (leaf_value_norm is ignored)
- **Skips** bottom-up value propagation (node_value_mode is ignored)
- **Skips** node advantage computation
- **Skips** token-level advantage mapping from tree structure
- **Instead** uses standard GRPO: `token_level_scores = reward_tensor`, then calls `compute_advantage()` with the GRPO estimator (group-relative outcome advantage)

### What stays the same

- Entropy-based branching still works (`expansion_mode=entropy`)
- Tree is still built with 10 initial rollouts, 2 expansion rounds, 22 sampled leaves
- Multi-turn tool calling still works
- GSPO-turn policy loss still works
- The tree structure provides **diverse trajectories** — just the advantage signal is flat (same advantage for all tokens in a trajectory, like vanilla GRPO)

### Config diff (only 1 change needed)

```diff
- actor_rollout_ref.rollout.node_adv_mode=node_value
+ actor_rollout_ref.rollout.node_adv_mode=vanilla
```

The `leaf_value_norm` and `node_value_mode` settings can stay as-is — they're simply ignored when `node_adv_mode=vanilla`.

### Why you might want this

- **Simpler baseline**: isolate the effect of entropy branching from credit assignment
- **Fewer moving parts**: standard GRPO advantage is well-understood
- **Faster debugging**: if training is unstable, rule out credit assignment as the cause

---

## Experiment Checklist for Ablation Studies

When running credit assignment ablations, change **one knob at a time** and compare:

1. **Vanilla baseline**: `node_adv_mode=vanilla` (disables tree credit assignment)
2. **Value propagation**: `child_softmax` vs `child_mean` vs `leaf_mean`
3. **Advantage mode**: `node_value` vs `diff_parent` vs `diff_global`
4. **Normalization**: `leaf_value_norm=True` vs `False`
5. **Policy loss**: `gspo_turn` vs `gspo` vs `grpo`
6. **Expansion strategy**: `entropy` vs `random`

For each ablation, monitor these W&B metrics:
- `val-core/*/acc/mean@*` — validation accuracy
- `actor/ppo_kl` — policy divergence from reference
- `actor/turn_kl_entropy` — how evenly updates distribute across turns
- `actor/pg_clipfrac` — fraction of clipped policy gradient updates
- `actor/entropy_loss` — model entropy (exploration)
