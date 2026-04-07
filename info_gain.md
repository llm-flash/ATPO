# Introduction
In this branch, vllm_rollout_with_tools_tree_offline_info_gain.py has been added which extends vllm_rollout_with_tools_tree_offline.py to calculate the information gain automatically and optionally use it instead of entropy for choosing which nodes to expand. The computation is integrated into _generate_chain_for_nodes, which calls the get_ground_truth_logprobs function. Before this happens, the pseudo-responses containing the ground-truth answers are formed using the get_pseudo_responses function.

**IMPORTANT** - You must set actor_rollout_ref.rollout.mode to "sync_with_tool_tree_info_gain" in order for the info gain code to be used, and you must set actor_rollout_ref.rollout.expansion_mode to "info_gain" for it to affect node expansion.

# Pseudo-response creation
The code assumes that the DataProto object for a training batch will have a 'reward_model' field in its non_tensor_batch which stores the ground-truth answers (during validation this will not be the case, so I avoid running any information gain computations during validation). Note that the code used to get the ground-truth answers from the data may need to change if we change the dataset/task; right now I have it hard-coded for the search task. The get_pseudo_responses function uses the ground-truth strings to create pseudo-responses and tokenize them. The indices of tokens corresponding to the actual answer are also saved; this information is stored in a tuple for each prompt. During tree creation, the root nodes are assigned this information and their child nodes inherit it from them; this makes the right pseudo-response easy to access for any given node.

# Ground-truth probability calculation
Any time a node which contains only a prompt or which ends in a tool result is created, its ground-truth answer probability is computed by appending the pseudo-response to its token ids, running a forward pass which returns the log probabilities for prompt tokens, and extracting the log probabilities corresponding to the indices of tokens corresponding to the actual answer. The forward pass is done using the same vllm inference engine that is used for the rollouts themselves.

# Using info gain for node expansion
In the sample_expansion_nodes function, if the selection mode is set to info_gain, the information gain of a node will be used in place of its entropy for deciding its score.

# Stats logged to wandb
The following statistics are computed across nodes on the same level and logged to wandb under the igpo tag:
- Average information gain
- Standard deviation of information gain
- Average ground-truth probability
Since these statistics are computed per-level, it is a good idea to create a single plot that combines all the levels so that you can compare between them. Additionally, the overall information gain (the difference between a leaf node's ground truth probability and the root node's ground-truth probability) averaged over all leaf nodes is recorded.

# Disclaimers
Currently we only consider one ground-truth answer when computing the ground-truth probability, so if multiple are given then only the first will actually be used. I have yet to encounter any problems in the training data that actually have multiple. Also, nodes which do not have any generated tokens are given a baseline score during node expansion since they do not have any info gain value. This could lead to suboptimal sampling behavior so it may need to be changed.