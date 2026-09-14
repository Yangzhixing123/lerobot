#!/usr/bin/env python

"""Evaluate decoded policy actions on recorded dataset frames, without a robot.

The loader supports the local PI0Fast and OAT checkpoints used by this repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import is_torch_device_available

DEFAULT_POLICY = "yangzhixing/pi0fast-base-lora-so101-box-to-plate"
DEFAULT_DATASET_REPO = "maxlium/so101-box-to-plate"
DEFAULT_DATASET_ROOT = "/home/zhixingyang/projects/so101-box-to-plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a policy frame-by-frame on a local LeRobot dataset.")
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY,
        help="Hugging Face repo ID or local PI0Fast LoRA checkpoint directory.",
    )
    parser.add_argument("--policy-revision", default=None)
    parser.add_argument(
        "--action-tokenizer-path",
        default=None,
        help="Local FAST tokenizer directory; avoids Hugging Face network access.",
    )
    parser.add_argument(
        "--text-tokenizer-path",
        default=None,
        help="Local PaliGemma tokenizer directory; avoids Hugging Face network access.",
    )

    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Optional episode; --start-frame then means an offset inside it.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Global index, or episode-local offset with --episode-index.",
    )
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=None,
        help="Evaluate this many evenly spaced frames from every episode.",
    )
    parser.add_argument("--task", default=None, help="Override the task recorded in each frame.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override policies with a temperature setting; use 0 for deterministic greedy decoding.",
    )
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    parser.add_argument("--print-full-chunk", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Suppress verbose per-frame output.")
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_frame_indices(
    meta: LeRobotDatasetMetadata,
    episode_index: int | None,
    start_frame: int,
    num_frames: int,
    stride: int,
) -> list[int]:
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    if num_frames < 1:
        raise ValueError("--num-frames must be positive.")
    if stride < 1:
        raise ValueError("--stride must be positive.")

    if episode_index is None:
        first = start_frame
        stop = meta.total_frames
    else:
        if not 0 <= episode_index < meta.total_episodes:
            raise ValueError(
                f"--episode-index must be in [0, {meta.total_episodes - 1}], got {episode_index}."
            )
        episode = meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_stop = int(episode["dataset_to_index"])
        first = episode_start + start_frame
        stop = episode_stop
        if first >= stop:
            raise ValueError(
                f"Episode {episode_index} has {stop - episode_start} frames; "
                f"--start-frame={start_frame} is outside it."
            )

    indices = [first + offset * stride for offset in range(num_frames)]
    indices = [index for index in indices if index < stop]
    if not indices:
        raise ValueError("The requested frame selection is empty.")
    return indices


def resolve_episode_sample_indices(meta: LeRobotDatasetMetadata, frames_per_episode: int) -> list[int]:
    if frames_per_episode < 1:
        raise ValueError("--frames-per-episode must be positive.")

    indices: list[int] = []
    for episode_index in range(meta.total_episodes):
        episode = meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        length = stop - start
        for sample_index in range(1, frames_per_episode + 1):
            offset = min(length - 1, sample_index * length // (frames_per_episode + 1))
            indices.append(start + offset)
    return indices


def to_float_images(frame: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    frame = dict(frame)
    for key in camera_keys:
        value = frame.get(key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            frame[key] = value.to(dtype=torch.float32) / 255.0
    return frame


def tensor_list(value: torch.Tensor, digits: int = 4) -> list[Any]:
    def round_nested(item: Any) -> Any:
        if isinstance(item, list):
            return [round_nested(child) for child in item]
        return round(float(item), digits)

    return round_nested(value.detach().cpu().tolist())


def scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def load_policy_and_processors(
    policy_path: str,
    policy_revision: str | None,
    device: str,
    dataset_meta: LeRobotDatasetMetadata,
    action_tokenizer_path: str | None,
    text_tokenizer_path: str | None,
) -> tuple[torch.nn.Module, Any, Any, PreTrainedConfig]:
    """Load a supported policy checkpoint and its saved processors."""
    config = PreTrainedConfig.from_pretrained(policy_path, revision=policy_revision)
    supported_types = {"pi0_fast", "oat_fsq", "oat_rfsq_pair"}
    if config.type not in supported_types:
        raise ValueError(
            f"Expected one of {sorted(supported_types)}, but {policy_path!r} contains type={config.type!r}."
        )

    # The adapter's saved config still points at the base model. Override it so
    # make_policy follows the PEFT path and attaches this trained LoRA adapter.
    config.pretrained_path = policy_path
    config.pretrained_revision = policy_revision
    config.device = device
    if hasattr(config, "gradient_checkpointing"):
        config.gradient_checkpointing = False
    if action_tokenizer_path is not None:
        config.action_tokenizer_name = str(Path(action_tokenizer_path).expanduser().resolve())
    if text_tokenizer_path is not None:
        config.text_tokenizer_name = str(Path(text_tokenizer_path).expanduser().resolve())

    policy = make_policy(config, ds_meta=dataset_meta)
    policy.eval()
    policy.requires_grad_(False)

    # These files carry the tokenizer setup and the dataset normalization stats.
    preprocessor_overrides = {"device_processor": {"device": device}}
    if action_tokenizer_path is not None:
        preprocessor_overrides["action_tokenizer_processor"] = {
            "action_tokenizer_name": config.action_tokenizer_name
        }
    if text_tokenizer_path is not None:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": config.text_tokenizer_name}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=policy_path,
        pretrained_revision=policy_revision,
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy, preprocessor, postprocessor, config


def evaluate_frame(
    *,
    dataset_index: int,
    frame: dict[str, Any],
    camera_keys: list[str],
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    task_override: str | None,
    print_full_chunk: bool,
) -> dict[str, Any]:
    raw_state = frame[OBS_STATE].to(dtype=torch.float32)
    recorded_actions = frame[ACTION].to(dtype=torch.float32)
    if recorded_actions.ndim == 1:
        recorded_actions = recorded_actions.unsqueeze(0)

    model_frame = to_float_images(frame, camera_keys)
    # Dataset temporal deltas are unbatched [T, ...]. The default processor only
    # recognizes non-temporal tensors, so add the single-sample batch explicitly.
    if raw_state.ndim > 1:
        model_frame[OBS_STATE] = model_frame[OBS_STATE].unsqueeze(0)
        for key in camera_keys:
            value = model_frame.get(key)
            if isinstance(value, torch.Tensor):
                model_frame[key] = value.unsqueeze(0)
    if task_override is not None:
        model_frame["task"] = task_override
    processed_frame = preprocessor(model_frame)

    policy.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        normalized_prediction = policy.predict_action_chunk(processed_frame)
        predicted_actions = postprocessor(normalized_prediction)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started_at

    if predicted_actions.ndim == 3:
        if predicted_actions.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(predicted_actions.shape)}.")
        predicted_actions = predicted_actions[0]
    if predicted_actions.ndim == 1:
        predicted_actions = predicted_actions.unsqueeze(0)

    horizon = min(predicted_actions.shape[0], recorded_actions.shape[0])
    action_dim = min(predicted_actions.shape[-1], recorded_actions.shape[-1])
    prediction = predicted_actions[:horizon, :action_dim].detach().cpu().float()
    target = recorded_actions[:horizon, :action_dim].detach().cpu().float()
    # Temporal policies receive state history as [T, D]; the final item is the
    # current state associated with the first predicted action.
    current_state = raw_state[-1] if raw_state.ndim > 1 else raw_state
    state = current_state[:action_dim].detach().cpu().float()
    error = prediction - target

    result: dict[str, Any] = {
        "dataset_index": dataset_index,
        "episode_index": scalar_int(frame["episode_index"]),
        "frame_index": scalar_int(frame["frame_index"]),
        "task": model_frame.get("task"),
        "inference_seconds": round(inference_seconds, 4),
        "state": tensor_list(state),
        "predicted_first_action": tensor_list(prediction[0]),
        "recorded_first_action": tensor_list(target[0]),
        "predicted_first_minus_state": tensor_list(prediction[0] - state),
        "first_action_abs_error": tensor_list(error[0].abs()),
        "first_action_mse": float(error[0].square().mean()),
        "first_action_mae": float(error[0].abs().mean()),
        "chunk_mse": float(error.square().mean()),
        "chunk_mae": float(error.abs().mean()),
        "per_joint_rmse": tensor_list(error.square().mean(dim=0).sqrt()),
        "per_joint_mse": tensor_list(error.square().mean(dim=0), digits=8),
        "per_joint_mae": tensor_list(error.abs().mean(dim=0)),
        "predicted_shape": list(predicted_actions.shape),
        "recorded_shape": list(recorded_actions.shape),
        "compared_shape": list(error.shape),
    }
    if print_full_chunk:
        result["predicted_chunk"] = tensor_list(prediction)
        result["recorded_chunk"] = tensor_list(target)
    return result


def print_frame_result(result: dict[str, Any], action_names: list[str]) -> None:
    print()
    print(
        f"[dataset={result['dataset_index']} episode={result['episode_index']} "
        f"frame={result['frame_index']}] task={result['task']!r}"
    )
    print(f"  inference:                 {result['inference_seconds']:.4f} s")
    print(f"  joint names:               {action_names}")
    print(f"  current state:             {result['state']}")
    print(f"  predicted first action:    {result['predicted_first_action']}")
    print(f"  recorded first action:     {result['recorded_first_action']}")
    print(f"  predicted action - state:  {result['predicted_first_minus_state']}")
    print(f"  first-action abs error:    {result['first_action_abs_error']}")
    print(f"  chunk MSE:                 {result['chunk_mse']:.6f}")
    print(f"  chunk MAE:                 {result['chunk_mae']:.6f}")
    print(f"  per-joint RMSE:            {result['per_joint_rmse']}")
    print(f"  compared chunk shape:      {result['compared_shape']}")
    if "predicted_chunk" in result:
        print(f"  predicted full chunk:      {result['predicted_chunk']}")
        print(f"  recorded full chunk:       {result['recorded_chunk']}")


def summarize_results(results: list[dict[str, Any]], action_names: list[str]) -> dict[str, Any]:
    result_count = len(results)
    action_dim = min(len(action_names), len(results[0]["per_joint_mse"]))
    per_joint_mse = [
        sum(float(result["per_joint_mse"][index]) for result in results) / result_count
        for index in range(action_dim)
    ]
    per_joint_mae = [
        sum(float(result["per_joint_mae"][index]) for result in results) / result_count
        for index in range(action_dim)
    ]

    episode_results: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        episode_results.setdefault(int(result["episode_index"]), []).append(result)
    per_episode = [
        {
            "episode_index": episode_index,
            "frames": len(items),
            "mean_chunk_mse": sum(item["chunk_mse"] for item in items) / len(items),
            "mean_chunk_mae": sum(item["chunk_mae"] for item in items) / len(items),
        }
        for episode_index, items in sorted(episode_results.items())
    ]
    worst_frames = sorted(results, key=lambda result: result["chunk_mse"], reverse=True)[:10]

    return {
        "successful_frames": result_count,
        "mean_chunk_mse": sum(result["chunk_mse"] for result in results) / result_count,
        "mean_chunk_mae": sum(result["chunk_mae"] for result in results) / result_count,
        "mean_first_action_mse": sum(result["first_action_mse"] for result in results) / result_count,
        "mean_first_action_mae": sum(result["first_action_mae"] for result in results) / result_count,
        "per_joint_rmse": [value**0.5 for value in per_joint_mse],
        "per_joint_mae": per_joint_mae,
        "per_episode": per_episode,
        "worst_frames": [
            {
                key: result[key]
                for key in ("dataset_index", "episode_index", "frame_index", "chunk_mse", "chunk_mae")
            }
            for result in worst_frames
        ],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not is_torch_device_available(args.device):
        raise RuntimeError(
            f"Requested --device={args.device!r}, but that device is not available. "
            "Use --device=cpu only for a slow smoke test."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Local dataset root does not exist: {dataset_root}")

    print(f"Reading dataset metadata from {dataset_root}")
    dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    if args.frames_per_episode is not None:
        frame_indices = resolve_episode_sample_indices(dataset_meta, args.frames_per_episode)
    else:
        frame_indices = resolve_frame_indices(
            dataset_meta,
            episode_index=args.episode_index,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            stride=args.stride,
        )

    print(f"Loading policy {args.policy_path!r} on {args.device} ...")
    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        policy_path=args.policy_path,
        policy_revision=args.policy_revision,
        action_tokenizer_path=args.action_tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        device=args.device,
        dataset_meta=dataset_meta,
    )
    if args.temperature is not None:
        if not hasattr(config, "temperature"):
            raise ValueError(f"Policy type {config.type!r} does not expose a temperature setting.")
        config.temperature = args.temperature
    print(
        f"Loaded {config.type}; PEFT={config.use_peft}; "
        f"prediction horizon={len(config.action_delta_indices)}; model wrapper={type(policy).__name__}"
    )

    delta_timestamps = resolve_delta_timestamps(config, dataset_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    action_feature = config.action_feature
    default_action_dim = action_feature.shape[0] if action_feature is not None else 0
    action_names = list(
        dataset.meta.features.get(ACTION, {}).get(
            "names", [f"joint_{index}" for index in range(default_action_dim)]
        )
    )

    if len(frame_indices) <= 20:
        selection = str(frame_indices)
    else:
        selection = f"{frame_indices[:5]} ... {frame_indices[-5:]}"
    print(f"Evaluating {len(frame_indices)} frame(s): {selection}; cameras={dataset.meta.camera_keys}")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_number, dataset_index in enumerate(frame_indices, start=1):
        try:
            result = evaluate_frame(
                dataset_index=dataset_index,
                frame=dataset[dataset_index],
                camera_keys=dataset.meta.camera_keys,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task_override=args.task,
                print_full_chunk=args.print_full_chunk,
            )
        except Exception as error:
            if args.fail_fast:
                raise
            logging.exception("Frame %s failed", dataset_index)
            failures.append({"dataset_index": dataset_index, "error": repr(error)})
            continue
        results.append(result)
        if args.summary_only:
            if sample_number % 10 == 0 or sample_number == len(frame_indices):
                print(f"Progress: {sample_number}/{len(frame_indices)} frames")
        else:
            print_frame_result(result, action_names)

    if results:
        summary = summarize_results(results, action_names)
        print()
        print(
            f"Summary: successful={len(results)}, failed={len(failures)}, "
            f"mean decoded-action MSE={summary['mean_chunk_mse']:.6f}, "
            f"MAE={summary['mean_chunk_mae']:.6f}"
        )
        print(f"Per-joint RMSE: {tensor_list(torch.tensor(summary['per_joint_rmse']))}")
        print(f"Per-joint MAE:  {tensor_list(torch.tensor(summary['per_joint_mae']))}")
        print(
            "Note: decoded-action MSE is in the dataset's physical action units; "
            "it is not numerically comparable to token cross-entropy loss."
        )
    else:
        raise RuntimeError(f"No frame was evaluated successfully. Failures: {failures}")

    if args.save_json is not None:
        output_path = args.save_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_path": args.policy_path,
            "policy_revision": args.policy_revision,
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": str(dataset_root),
            "device": args.device,
            "temperature": getattr(config, "temperature", None),
            "action_names": action_names,
            "sampling": {
                "frames_per_episode": args.frames_per_episode,
                "total_episodes": dataset_meta.total_episodes,
                "requested_frames": len(frame_indices),
            },
            "summary": summary,
            "results": results,
            "failures": failures,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
