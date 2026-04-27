# Introduction
In this branch, vllm_rollout_with_tools_tree_offline_cache.py has been added which extends vllm_rollout_with_tools_tree_offline.py to calculate curiosity values and use them to apply an advantage bonus. The computation is integrated into _generate_chain_for_nodes, which calls the get_ground_truth_logprobs function. Before this happens, the pseudo-responses containing the ground-truth answers are formed using the get_pseudo_responses function.

**IMPORTANT** - You must set actor_rollout_ref.rollout.mode to "sync_with_tool_tree_cache" in order for the curiosity bonus to be used.

# Modifications to the Advantage Calculation
By default, the advantage of a node is computed as V' + beta * C', where V' is the normalized value, beta is an annealed learning rate (starting at 1.3 and decaying to 0.6), and C' is the normalized curiosity. The raw curiosity is defined as V - P(ans) where V is the unnormalized value and P(ans) is the ground-truth probability at that node (i.e. the probability assigned to the ground-truth answer conditioned on all the context up to and including that node). You can include entropy in the bonus by setting actor_rollout_ref.rollout.entropy_mixing_method to "multiplicative" or "additive". In that case, the computation will be V' + beta * H * C' (for multiplicative) or V' + beta_1 * H + beta_2 * C' (for additive). Note that beta_1 and beta_2 will be identical by default, and you must modify the code if you want to make them different. You can change the number of steps over which beta is annealed by setting actor_rollout_ref.rollout.annealing_steps (the default is 150). By default, a cosine annealing schedule will be used; this can be replaced with a monotonically decreasing schedule by setting actor_rollout_ref.rollout.cosine_annealing to false.

# Pseudo-response creation
The code assumes that the DataProto object for a training batch will have a 'reward_model' field in its non_tensor_batch which stores the ground-truth answers (during validation this will not be the case, so I avoid running any information gain computations during validation). Note that the code used to get the ground-truth answers from the data may need to change if we change the dataset/task; right now I have it hard-coded for the search task. The get_pseudo_responses function uses the ground-truth strings to create pseudo-responses and tokenize them. The indices of tokens corresponding to the actual answer are also saved; this information is stored in a tuple for each prompt. During tree creation, the root nodes are assigned this information and their child nodes inherit it from them; this makes the right pseudo-response easy to access for any given node.

# Ground-truth probability calculation
Any time a node which contains only a prompt or which ends in a tool result is created, its ground-truth answer probability is computed by appending the pseudo-response to its token ids, running a forward pass which returns the log probabilities for prompt tokens, and extracting the log probabilities corresponding to the indices of tokens corresponding to the actual answer. The forward pass is done using the same vllm inference engine that is used for the rollouts themselves.


# Stats logged to wandb
The following statistics are computed across nodes on the same level and logged to wandb under the igpo tag:
- Average information gain
- Standard deviation of information gain
- Average ground-truth probability
- Average entropy and curiosity
Since these statistics are computed per-level, it is a good idea to create a single plot that combines all the levels so that you can compare between them. Additionally, the overall information gain (the difference between a leaf node's ground truth probability and the root node's ground-truth probability) averaged over all leaf nodes is recorded. Furthermore, the correlation between ground-truth probability and node value is recorded.

# Disclaimers
Currently we only consider one ground-truth answer when computing the ground-truth probability, so if multiple are given then only the first will actually be used. I have yet to encounter any problems in the training data that actually have multiple.