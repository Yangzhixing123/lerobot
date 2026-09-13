#!/usr/bin/env python
"""PI0-Fast LoRA async policy server.

This folder is intentionally isolated from ``src/lerobot``. It reuses the
upstream async gRPC protocol, but adds the PEFT loading path needed by
``yangzhixing/pi0fast-base-lora-so101-box-to-plate``.
"""

import json
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import grpc
import torch
from peft import PeftConfig, PeftModel

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    Observation,
    RawObservation,
    TimedObservation,
    extract_images_from_raw_observation,
    extract_state_from_raw_observation,
    is_image_key,
    make_lerobot_observation,
    prepare_image,
    resize_robot_observation_image,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs import PolicyFeature, PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging


@dataclass
class PI0FastPolicyServerConfig(PolicyServerConfig):
    cache_dir: str = field(
        default="/data/ywb-2/yzx/lerobot/.cache/huggingface/hub",
        metadata={"help": "Hugging Face hub cache directory used by the Linux policy server."},
    )
    log_dir: str = field(
        default="/data/ywb-2/yzx/lerobot/pi0fast_async/logs/server",
        metadata={"help": "Directory for server logs and JSONL telemetry."},
    )
    strict_load: bool = field(
        default=False,
        metadata={"help": "Whether to strictly load the PI0-Fast base safetensors."},
    )


class JsonlTelemetry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **payload: Any) -> None:
        record = {"time": time.time(), "event": event, **payload}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _apply_rename_map(obs: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    if not rename_map:
        return obs
    return {rename_map.get(key, key): value for key, value in obs.items()}


def raw_observation_to_observation_with_rename(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    rename_map: dict[str, str],
) -> Observation:
    lerobot_obs = make_lerobot_observation(raw_observation, lerobot_features)
    lerobot_obs = _apply_rename_map(lerobot_obs, rename_map)

    image_keys = list(filter(is_image_key, lerobot_obs))
    missing = [key for key in image_keys if key not in policy_image_features]
    if missing:
        raise KeyError(
            "Observation image keys are not present in policy image features after rename_map. "
            f"missing={missing}, available={list(policy_image_features)}, rename_map={rename_map}"
        )

    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        key: resize_robot_observation_image(
            torch.tensor(extract_images_from_raw_observation(lerobot_obs, key)),
            policy_image_features[key].shape,
        )
        for key in image_keys
    }

    if "task" in raw_observation:
        state_dict["task"] = raw_observation["task"]

    observation = {**state_dict, **image_dict}
    for key, value in list(observation.items()):
        if isinstance(value, torch.Tensor) and "image" in key:
            observation[key] = prepare_image(value).unsqueeze(0)
    return observation


class PI0FastPolicyServer(PolicyServer):
    def __init__(self, config: PI0FastPolicyServerConfig):
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        init_logging(log_file=log_dir / f"policy_server_{int(time.time())}.log", display_pid=False)
        super().__init__(config)
        self.config: PI0FastPolicyServerConfig
        self.rename_map: dict[str, str] = {}
        self.loaded_policy_signature: dict[str, Any] | None = None
        self.telemetry = JsonlTelemetry(log_dir / f"policy_server_telemetry_{int(time.time())}.jsonl")
        self.telemetry.write(
            "server_init",
            host=config.host,
            port=config.port,
            fps=config.fps,
            inference_latency=config.inference_latency,
            obs_queue_timeout=config.obs_queue_timeout,
            cache_dir=config.cache_dir,
        )

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _load_pi0fast_peft_policy(self, adapter_path: str, device: str):
        cache_dir = self.config.cache_dir
        adapter_cfg = PreTrainedConfig.from_pretrained(adapter_path, cache_dir=cache_dir)
        if adapter_cfg.type != "pi0_fast":
            raise ValueError(f"Expected a pi0_fast adapter config, got {adapter_cfg.type!r}")

        adapter_cfg.device = device
        peft_config = PeftConfig.from_pretrained(adapter_path, cache_dir=cache_dir)
        base_model = peft_config.base_model_name_or_path or adapter_cfg.pretrained_path
        if not base_model:
            raise ValueError("Adapter config has no base_model_name_or_path or pretrained_path.")

        policy_class = get_policy_class(adapter_cfg.type)
        self.logger.info("Loading PI0-Fast base policy %s for adapter %s", base_model, adapter_path)
        policy = policy_class.from_pretrained(
            base_model,
            config=adapter_cfg,
            cache_dir=cache_dir,
            token=True,
            strict=self.config.strict_load,
        )
        policy = PeftModel.from_pretrained(
            policy,
            adapter_path,
            config=peft_config,
            cache_dir=cache_dir,
            token=True,
            is_trainable=False,
        )
        policy.to(device)
        policy.eval()
        return policy, adapter_cfg, base_model

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec
        self.rename_map = dict(getattr(policy_specs, "rename_map", {}))

        if policy_specs.policy_type != "pi0_fast":
            raise ValueError("This isolated server only supports --policy_type=pi0_fast")

        signature = {
            "policy_type": policy_specs.policy_type,
            "pretrained_name_or_path": policy_specs.pretrained_name_or_path,
            "actions_per_chunk": policy_specs.actions_per_chunk,
            "device": policy_specs.device,
            "rename_map": self.rename_map,
        }
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.device = policy_specs.device

        if self.policy is not None and self.loaded_policy_signature == signature:
            self.logger.info("Reusing already loaded PI0-Fast adapter: %s", policy_specs.pretrained_name_or_path)
            self.telemetry.write("policy_reused", **signature)
            return services_pb2.Empty()

        start = time.perf_counter()
        self.policy, policy_cfg, base_model = self._load_pi0fast_peft_policy(
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
        )

        device_override = {"device": policy_specs.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": self.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        elapsed = time.perf_counter() - start
        self.loaded_policy_signature = signature
        self.logger.info(
            "Loaded PI0-Fast adapter on %s in %.2fs | adapter=%s | base=%s",
            policy_specs.device,
            elapsed,
            policy_specs.pretrained_name_or_path,
            base_model,
        )
        self.telemetry.write("policy_loaded", elapsed_s=elapsed, base_model=base_model, **signature)
        return services_pb2.Empty()

    def _predict_action_chunk(self, observation_t: TimedObservation):
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation_with_rename(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.rename_map,
        )
        prepare_time = time.perf_counter() - start_prepare

        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference

        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            processed_actions.append(self.postprocessor(action_tensor[:, i, :]))
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()
        postprocessing_time = time.perf_counter() - start_postprocess

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        total_time = prepare_time + preprocessing_time + inference_time + postprocessing_time
        self.telemetry.write(
            "action_chunk",
            observation_timestep=observation_t.get_timestep(),
            chunk_size=len(action_chunk),
            prepare_ms=prepare_time * 1000,
            preprocess_ms=preprocessing_time * 1000,
            inference_ms=inference_time * 1000,
            postprocess_ms=postprocessing_time * 1000,
            total_ms=total_time * 1000,
        )
        self.logger.info(
            "Observation %s | action chunk=%s | total=%.2fms | inference=%.2fms",
            observation_t.get_timestep(),
            len(action_chunk),
            total_time * 1000,
            inference_time * 1000,
        )
        return action_chunk


@draccus.wrap()
def serve(cfg: PI0FastPolicyServerConfig):
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    policy_server = PI0FastPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info("PI0-Fast PolicyServer started on %s:%s", cfg.host, cfg.port)
    policy_server.telemetry.write("server_started", host=cfg.host, port=cfg.port)
    server.start()
    try:
        server.wait_for_termination()
    finally:
        policy_server.telemetry.write("server_terminated")


if __name__ == "__main__":
    serve()
