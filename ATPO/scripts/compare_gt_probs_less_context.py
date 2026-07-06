from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoConfig

os.environ["VERL_LOGGING_LEVEL"] = "WARNING"

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl_atpo"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl import DataProto  # noqa: E402
from verl.utils.model import compute_position_id_with_mask  # noqa: E402
from verl.utils.reward_score.search_r1_like_qa_em import extract_solution, em_check  # noqa: E402
from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed  # noqa: E402
from verl.utils.tokenizer import hf_tokenizer  # noqa: E402
from verl.utils.torch_functional import postprocess_data  # noqa: E402
from verl.workers.rollout.vllm_rollout.vllm_rollout_with_tools_tree_offline_cache import (  # noqa: E402
    vLLMRolloutWithTools,
)

# NOTE: This class is just for safety during testing, it can be removed later
class DirectvLLMRolloutWithTools(vLLMRolloutWithTools):
    """Small wrapper so failed init does not hide the real exception."""

    def __init__(self, *args, **kwargs):
        self.executor = None
        super().__init__(*args, **kwargs)

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B", help="HF model path or local checkpoint.")
    parser.add_argument("--prompts-file", help="Optional .jsonl/.json/.txt prompt file.")
    parser.add_argument("--num_prompts", type=int, default=None, help="Optional limit on number of prompts to process.")
    parser.add_argument("--output-file", default="ATPO/scripts/direct_then_modified_outputs.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=8000) # Because we are not starting from the original prompt, but rather a truncated version of the full response
    parser.add_argument("--response-length", type=int, default=192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--breakdown-by-id",
        action="store_true",
        help="Print per-sample statistics keyed by output id.",
    )
    return parser.parse_args()


def maybe_init_distributed() -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            init_method="env://" # TODO: Can try this
        )


def build_rollout_config(args: argparse.Namespace):
    samples_per_tree = 1
    return OmegaConf.create(
        {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "name": "vllm",
            "mode": "sync_with_tool_tree_cache",
            "chat_scheduler": None,
            "use_fire_sampling": False,
            "initial_rollouts": 1,
            "expansion_mode": "entropy",
            "expansion_iterations": 0,
            "beam_size": 1,
            "samples_per_tree": samples_per_tree,
            "n": samples_per_tree,
            "branch_probability": 0.0,
            "entropy_weight": 0.2,
            "data": {"reward_fn_key": "data_source"},
            "reward_model": {"reward_manager": "naive", "reward_kwargs": {}},
            "custom_reward_function": None,
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "norm_adv_by_std_in_grpo": True,
            "kl_ctrl": {},
            "leaf_value_norm": True,
            "node_value_mode": "child_mean",
            "node_adv_mode": "node_value",
            "debug_checks": True,
            "annealing_steps": 150,
            "cosine_annealing": True,
            "entropy_mixing_method": "no_entropy",
            "enable_dynamic_rollouts": False,
            # "multi_turn": {
            #     "enable": True,
            #     "max_turns": 3,
            #     "tool_config_path": str(REPO_ROOT / "verl_atpo/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"),
            #     "format": "chatml",
            # },
            "tools": {
                "call_limit": 0,
                "max_workers": 32,
                "timeout": 120,
                "retry_count": 4,
                "verbose_logging": True,
                "fail_on_error": False,
                "tool_instances": {},
            },
            "gt_prob": {"debug_checks": False},
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "ignore_eos": False,
            "enforce_eager": True,
            "free_cache_engine": False,
            "load_format": "auto",
            "layered_summon": False,
            "tensor_model_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": 25000,
            "max_model_len": args.prompt_length + args.response_length,
            "max_num_seqs": 1024,
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": None,
            "log_prob_use_dynamic_bsz": True,
            "log_prob_max_token_len_per_gpu": 4 * (args.prompt_length + args.response_length),
            "disable_log_stats": True,
            "enable_chunked_prefill": False,
            "do_sample": True,
            "seed": 0,
            "limit_images": None,
            "engine_kwargs": {"vllm": {"swap_space": None}, "sglang": {"attention_backend": None}},
            "val_kwargs": {
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "n": 1,
                "do_sample": False,
            },
        }
    )


def build_rollout_device_mesh(tensor_parallel_size: int):
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by "
            f"--tensor-parallel-size={tensor_parallel_size}"
        )
    dp = world_size // tensor_parallel_size
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_name,
        mesh_shape=(dp, tensor_parallel_size),
        mesh_dim_names=["dp", "infer_tp"],
    )


def dummy_hotpotqa_prompt() -> dict[str, Any]:
    return {
        "prompt": "system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. You can search as many times as your want. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result>  <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.\nuser\nWere Scott Derrickson and Ed Wood of the same nationality?\nassistant\n<think>\nOkay, let's see. The user is asking if Scott Derrickson and Ed Wood were of the same nationality. I need to figure out the nationalities of both individuals and compare them.\n\nFirst, I should start by recalling who these people are. Scott Derrickson is a filmmaker, I think. He's known for directing movies like \"X-Men\" and \"The Exorcist: The Beginning.\" Ed Wood was a filmmaker as well, but I remember he's often associated with low-budget films and is sometimes referred to as the \"King of the Indies.\" But I'm not entirely sure about his nationality. \n\nLet me start by checking Scott Derrickson's nationality. I can search for Scott Derrickson's background. Maybe he's American. But I should confirm. Let me do a quick search.\n\n<search>Scott Derrickson nationality</search>\n <result>\nPage 1: \"Scott Derrickson\"\ndirected the film \"\"Doctor Strange\"\", based on the Marvel Comics property and part of the Marvel Cinematic Universe. It was released in November 2016. The film was both a commercial and critical success. In February 2018, it was announced that Derrickson will executive produce the television series \"\"You Bury Me\"\" for Blumhouse Productions. The series is set in Iraq, Syria, and Turkey, and will be a love story in contemporary war-torn Middle East. In December 2018, it was announced that Derrickson will direct the \"\"Doctor Strange\"\" sequel with a planned May 2021 release. Derrickson married a nurse in September 1993\nPage 2: \"Scott Tucker (racing driver)\"\n(in the TA2 class) in August 2011 at Road America. Tucker is the founder and CEO of Sentient Technologies, a provider of loan origination software. Tucker grew up in Kansas City, Missouri and went to Rockhurst High School. He studied business administration at Kansas State University. Tucker is married to his wife Kim and has two daughters. On January 5, 2018, Tucker was convicted of 14 criminal counts relating to his payday lending businesses and was sentenced to 16 years and 8 months in prison. Tucker is presently in the custody of the U.S. Marshals Service at an undisclosed location.\nPage 3: \"Scott Smith (ice hockey)\"\ncollaborated regularly with team management of Sean Burke with the men's national team, and Melody Davidson with the women's national team to ensure that staff and athletes were prepared to perform on the ice. In 2018, Smith extended the partnership with Bauer Hockey for another 8 years, who have been the official equipment supplier for the national teams since 1996. Smith was born November 27, 1966, in Bathurst, New Brunswick. He currently resides in Calgary, and is married to Karen, with one son named Jackson, who grew up playing hockey. Smith has family in the Bathurst area, including in-laws, and\n</result> \n\nHmm, the first result mentions Scott Derrickson as a director, but it doesn't state his nationality. However, given that he's a filmmaker in the US, and considering that many American directors are from the US, I might assume he's American. But I need to be sure. Let me check another source.\n\nAlternatively, maybe I can look up Ed Wood's nationality. Ed Wood is a well-known figure in film history, often associated with the 1950s. He was an American filmmaker, but I think he was from the United States. However, I should confirm.\n\n<search>Ed Wood nationality</search>\n <result>\nPage 1: \"Ed Wood\"\nGrey's 1992 oral biography \"\"Nightmare of Ecstasy: The Life and Art of Edward D. Wood Jr.\"\", the public took a greater interest in Wood, leading up to director Tim Burton's biopic of Wood's life, \"\"Ed Wood\"\" (1994), a critically acclaimed film starring Johnny Depp as Wood that earned two Academy Awards. Wood's father, Edward Sr., worked for the U.S. Postal Service as a custodian, and his family relocated numerous times around the United States. Eventually, they settled in Poughkeepsie, New York, where Ed Wood Jr. was born in 1924. According to Wood's second wife, Kathy O'Hara, Wood's mother Lillian would\nPage 2: \"Ed Wood\"\nEd Wood Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, and author. In the 1950s, Wood made a number of low-budget films in the science fiction, crime and horror genres, intercutting stock footage. In the 1960s and 1970s, he made sexploitation movies and wrote over 80 pulp crime, horror and sex novels. In 1975, he was awarded a Golden Turkey Award as Worst Director of All Time, renewing public interest in his work. Wood's career and camp approach has earned him and his films a cult following. Following the publication of Rudolph\nPage 3: \"Alexander Wood (soccer)\"\nAlexander Wood (soccer) Alexander “Alec” Wood (June 12, 1907 – July 20, 1987) Was a former Scottish American soccer defender. Wood began his club career in the United States before moving to England in the early 1930s. He also played all three U.S. games at the 1930 FIFA World Cup. He is a member of the National Soccer Hall of Fame. Wood’s parents moved the family to the United States in 1921 when Wood was fourteen years old. His family settled in Gary, Indiana where he attended Emerson High School and gained his U.S. citizenship a year later. He also\n</result> \n\nFrom the search results, Ed Wood is clearly identified as American. The first result states he was born in Poughkeepsie, New York, and the second result also confirms he was an American filmmaker. \n\nNow, going back to Scott Derrickson. The first search result for Scott Derrickson didn't mention nationality, but given that he's a director associated with Marvel and has worked in the US, it's likely he's American. However, to be thorough, I should check another source.\n\n<search>Scott Derrickson birthplace</search>\n <result>\nPage 1: \"Scott Beason\"\nBeason was born in Hartselle, Alabama. In 1991, he received a degree in geology from the University of Alabama at Tuscaloosa. He was a geologist from 1993 to 1997, having been employed by Bondurant Environmental Consulting. He currently owns Custom Renovators and Old South Construction. In 1994, he ran for Alabama's Senate District 17 as a Republican. He lost in the primary to incumbent State Representative Jack Biddle 60 to 40 percent. In 1998, won the primary runoff in Alabama's House District 51 against State Representative Jim Townsend with 53% of the vote. He won the general election with 99%\nPage 2: \"Scott Tucker (racing driver)\"\n(in the TA2 class) in August 2011 at Road America. Tucker is the founder and CEO of Sentient Technologies, a provider of loan origination software. Tucker grew up in Kansas City, Missouri and went to Rockhurst High School. He studied business administration at Kansas State University. Tucker is married to his wife Kim and has two daughters. On January 5, 2018, Tucker was convicted of 14 criminal counts relating to his payday lending businesses and was sentenced to 16 years and 8 months in prison. Tucker is presently in the custody of the U.S. Marshals Service at an undisclosed location.\nPage 3: \"Scott Dawson (evangelist)\"\nScott Dawson (evangelist) Scott Dawson (born September 14, 1967) is an American author, preacher, and the founder of the Scott Dawson Evangelistic Association (SDEA), and of the Strength to Stand Conference movement. Dawson was born on September 14, 1967. He graduated from Samford University earning a Bachelor of Arts, and earned a Master of Divinity from Beeson Divinity School in Birmingham, Alabama. Later in 1993, he received his ordination and certificate of gospel ministry by order of the Roebuck Park Baptist Church. He preached his first sermon at the age of 16. Shortly after, he began speaking at various churches.\n</result>",
        "ground_truth": "Yes",
    }

def load_prompt_records(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dummy_hotpotqa_prompt()]

    prompt_path = Path(path)
    if prompt_path.suffix == ".jsonl":
        with prompt_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if prompt_path.suffix == ".json":
        with prompt_path.open() as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    with prompt_path.open() as f:
        return [{"prompt": line.strip()} for i, line in enumerate(f) if line.strip()]


def split_records_by_truncation_point(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each prompt into variants ending at assistant/result boundaries."""
    split_records: list[dict[str, Any]] = []
    assistant_sep = "assistant\n"
    result_tag = "</result>"

    for source_index, record in enumerate(records):
        prompt = record.get("prompt", "")
        assistant_start = prompt.find(assistant_sep)

        if assistant_start == -1:
            truncated = deepcopy(record)
            truncated["source_record_id"] = record.get("id", source_index)
            truncated["truncation_point"] = None
            split_records.append(truncated)
            continue

        cut_points = [assistant_start + len(assistant_sep)]
        search_from = cut_points[0]
        while True:
            result_start = prompt.find(result_tag, search_from)
            if result_start == -1:
                break
            cut_points.append(result_start + len(result_tag))
            search_from = result_start + len(result_tag)

        for truncation_point, cut_point in enumerate(cut_points, start=1):
            truncated = deepcopy(record)
            truncated["prompt"] = prompt[:cut_point]
            truncated["source_record_id"] = record.get("id", source_index)
            truncated["truncation_point"] = truncation_point
            split_records.append(truncated)

    return split_records


def modify_prompt(record: dict[str, Any], direct_answer: str | None = None, direct_response: str | None = None) -> dict[str, Any]:
    """Remove all but the final <search>/<result> blocks from the assistant section.

    The system prompt (everything before the first 'assistant\\n') is preserved
    verbatim even if it contains <search>/<result> tags.  Only the part after
    the system prompt is modified.
    """
    modified = deepcopy(record)
    prompt = modified.get("prompt", "")

    # Split on "assistant\n" to separate system prompt from assistant content.
    sep = "assistant\n"
    if sep not in prompt:
        modified["changed"] = 0
        return modified

    sys_part, after_sys = prompt.split(sep, maxsplit=1)
    after_sys = sep + after_sys  # keep the separator

    # Find all <search>...</result> blocks (non-greedy per block).
    pattern = r"<search>.*?</search>\s*<result>.*?</result>"
    matches = list(re.finditer(pattern, after_sys, re.DOTALL))

    if len(matches) <= 1:
        # Nothing to remove.
        modified["changed"] = 0
        return modified

    # Keep only the last block; remove everything before it that matches.
    before_first_match = after_sys[: matches[0].start()]
    stuff_to_keep_inbetween = "".join(after_sys[matches[i].end():matches[i+1].start()] for i in range(len(matches)-1))
    new_after_sys = sys_part + before_first_match + stuff_to_keep_inbetween + after_sys[matches[-1].start():]
    modified["prompt"] = new_after_sys
    modified["changed"] = int(modified["prompt"] != prompt)
    return modified


def build_dataproto(records: list[dict[str, Any]], tokenizer, max_prompt_length: int) -> DataProto:
    input_ids_list = []
    attention_mask_list = []
    position_ids_list = []
    raw_prompt_ids = []

    max_record_length = max(len(tokenizer.encode(record["prompt"], add_special_tokens=False)) for record in records)

    for record in records:
        prompt = record["prompt"]
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = postprocess_data(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_length=max_record_length, #max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="left",
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        input_ids_list.append(input_ids[0])
        attention_mask_list.append(attention_mask[0])
        position_ids_list.append(position_ids[0])
        raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
        raw_prompt_ids.append(raw_ids[-max_prompt_length:])

    batch = TensorDict(
        {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "position_ids": torch.stack(position_ids_list),
        },
        batch_size=[len(records)],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)},
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False,
            "validate": False,
        },
    )

def get_gt_probs_batch(
    rollout: vLLMRolloutWithTools,
    tokenizer,
    records: list[dict[str, Any]],
    max_prompt_length: int,
    timings: list[float] | None = None,
) -> list[dict[str, Any]]:
    import time
    start = time.time()
    data = build_dataproto(records, tokenizer, max_prompt_length)
    data.non_tensor_batch['reward_model'] = [{'ground_truth': {'target': record["ground_truth"] if isinstance(record["ground_truth"], list) else [record["ground_truth"]]}} for record in records]
    pseudo_resp_info = rollout.get_pseudo_responses(data)
    elapsed = time.time() - start
    if timings is not None:
        timings.append(elapsed)
    pseudo_responses = [info[0] for info in pseudo_resp_info]
    gt_token_indices = [info[1] for info in pseudo_resp_info]
    gt_probs = rollout.get_ground_truth_probs(data.batch['input_ids'], pseudo_responses, gt_token_indices)
    results = []
    for i, record in enumerate(records):
        results.append(
            {
                "id": i,
                "pseudo_response": pseudo_responses[i],
                "ground_truth_prob": gt_probs[i],
            }
        )
    return results

def _print_timing_stats(direct_timings: list[float], modified_timings: list[float]) -> None:
    """Print aggregated timing statistics for direct and modified prompts."""
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)

    for name, timings in [("Direct", direct_timings), ("Modified", modified_timings)]:
        if not timings:
            print(f"{name}: no calls recorded")
            continue
        avg = sum(timings) / len(timings)
        mn = min(timings)
        mx = max(timings)
        print(f"{name} ({len(timings)} calls): avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    maybe_init_distributed()

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=args.trust_remote_code)
    model_hf_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )
    model_hf_config.bos_token_id = tokenizer.bos_token_id
    model_hf_config.eos_token_id = tokenizer.eos_token_id
    model_hf_config.pad_token_id = tokenizer.pad_token_id

    rollout = DirectvLLMRolloutWithTools(
        model_path=args.model_path,
        config=build_rollout_config(args),
        tokenizer=tokenizer,
        model_hf_config=model_hf_config,
        device_mesh=build_rollout_device_mesh(args.tensor_parallel_size),
        trust_remote_code=args.trust_remote_code,
    )
    if "tags" in inspect.signature(rollout.inference_engine.wake_up).parameters:
        rollout.inference_engine.wake_up(tags=["weights"])
        rollout.inference_engine.wake_up(tags=["kv_cache"])
    else:
        rollout.inference_engine.wake_up()

    records = split_records_by_truncation_point(load_prompt_records(args.prompts_file)[: args.num_prompts])
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direct_timings: list[float] = []
    modified_timings: list[float] = []

    with output_path.open("w") as f:
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            direct = get_gt_probs_batch(rollout, tokenizer, batch_records, args.prompt_length, timings=direct_timings)
            modified_records = [
                modify_prompt(record)
                for record in batch_records
            ]
            modified = get_gt_probs_batch(rollout, tokenizer, modified_records, args.prompt_length, timings=modified_timings)

            for i, (record, direct_result, modified_record, modified_result) in enumerate(zip(
                batch_records, direct, modified_records, modified
            )):
                f.write(
                    json.dumps(
                        {
                            "id": start + i,
                            "source_record_id": record.get("source_record_id", start + i),
                            "truncation_point": record.get("truncation_point"),
                            "original_prompt": record["prompt"],
                            "modified_prompt": modified_record["prompt"],
                            "gt_probs": {"direct": direct_result["ground_truth_prob"], "modified": modified_result["ground_truth_prob"]},
                            "changed": modified_record.get("changed", 0),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    # Compute and print statistics
    _print_statistics(output_path, breakdown_by_id=args.breakdown_by_id)

    # Print aggregated timing stats
    _print_timing_stats(direct_timings, modified_timings)


def _print_statistics(output_path: Path, breakdown_by_id: bool = False) -> None:
    """Read the output JSONL and compare direct vs. modified gt_probs."""
    total_rows = 0
    total_changed = 0
    total_direct_prob = 0.0
    total_modified_prob = 0.0
    total_diff = 0.0
    per_id_rows: list[dict[str, Any]] = []
    by_truncation_point: dict[Any, dict[str, float]] = {}

    with output_path.open() as f:
        for total, line in enumerate(f):
            rec = json.loads(line)
            sample_id = rec.get("id", total)
            truncation_point = rec.get("truncation_point")
            stats = by_truncation_point.setdefault(
                truncation_point,
                {"rows": 0, "changed": 0, "direct": 0.0, "modified": 0.0, "diff": 0.0},
            )
            stats["rows"] += 1
            total_rows += 1

            gt_probs = rec.get("gt_probs", {})
            direct_prob = float(gt_probs.get("direct", 0.0))
            modified_prob = float(gt_probs.get("modified", 0.0))
            diff = modified_prob - direct_prob
            stats['direct'] += direct_prob
            stats['modified'] += modified_prob
            stats['diff'] += diff

            if not rec.get("changed", 0):
                continue

            total_changed += 1
            total_direct_prob += direct_prob
            total_modified_prob += modified_prob
            total_diff += diff
            stats["changed"] += 1

            if breakdown_by_id:
                per_id_rows.append(
                    {
                        "id": sample_id,
                        "source_record_id": rec.get("source_record_id"),
                        "truncation_point": truncation_point,
                        "direct_prob": direct_prob,
                        "modified_prob": modified_prob,
                        "diff": diff,
                    }
                )

    if total_rows == 0:
        print("No samples to compute statistics.")
        return

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Samples: {total_rows}")
    print(f"Changed samples: {total_changed}")
    if total_changed > 0:
        print(f"Average direct gt_prob: {total_direct_prob / total_changed}")
        print(f"Average modified gt_prob: {total_modified_prob / total_changed}")
        print(f"Average difference (modified - direct): {total_diff / total_changed}")
    else:
        print("No changed samples to compute direct/modified averages.")

    print("\nBY TRUNCATION POINT")
    print("=" * 60)
    for truncation_point in sorted(by_truncation_point, key=lambda point: (point is None, point)):
        stats = by_truncation_point[truncation_point]
        row_count = int(stats["rows"])
        changed = int(stats["changed"])
        label = "None" if truncation_point is None else str(truncation_point)
        print(
            f"truncation_point={label}: samples={row_count}, changed={changed}, "
            f"direct={stats['direct'] / row_count}, modified={stats['modified'] / row_count}, "
            f"diff={stats['diff'] / row_count}"
        )

    if breakdown_by_id:
        print("\nBY ID")
        print("=" * 60)
        for row in per_id_rows:
            print(
                f"id={row['id']} source_record_id={row['source_record_id']} "
                f"truncation_point={row['truncation_point']}: direct={row['direct_prob']}, "
                f"modified={row['modified_prob']}, diff={row['diff']}"
            )
    print("=" * 60 + "\n")

# NOTE: This function is actually necessary to avoid a raw_delete error from CUDAPluggableAllocator.cpp
def run_and_exit() -> None:
    try:
        main()
    except BaseException:
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    run_and_exit()
