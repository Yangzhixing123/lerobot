#!/usr/bin/env python
"""Windows-side SO101 client for the PI0-Fast async policy server."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class PI0FastRobotClientConfig(RobotClientConfig):
    duration_s: float = field(default=5.0, metadata={"help": "Bounded robot run duration in seconds."})
    rename_map: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Optional observation-key rename map sent to the Linux server. "
            "Default is empty because this PI0-Fast SO101 policy expects front/wrist keys."
        },
    )
    log_dir: str = field(
        default="pi0fast_async/logs/client",
        metadata={"help": "Windows-side JSONL/log output directory."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.duration_s <= 0:
            raise ValueError("--duration_s must be positive")


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


class PI0FastRobotClient(RobotClient):
    def __init__(self, config: PI0FastRobotClientConfig):
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        init_logging(log_file=log_dir / f"robot_client_{int(time.time())}.log", display_pid=False)
        super().__init__(config)
        self.policy_config.rename_map = dict(config.rename_map)
        self.telemetry = JsonlTelemetry(log_dir / f"robot_client_telemetry_{int(time.time())}.jsonl")
        self.sent_observations = 0
        self.action_chunks_received = 0
        self.actions_received = 0
        self.actions_executed = 0
        self.telemetry.write(
            "client_init",
            server_address=config.server_address,
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            actions_per_chunk=config.actions_per_chunk,
            fps=config.fps,
            duration_s=config.duration_s,
            rename_map=config.rename_map,
            disable_torque_on_disconnect=getattr(config.robot, "disable_torque_on_disconnect", None),
        )

    def send_observation(self, obs) -> bool:
        start = time.perf_counter()
        ok = super().send_observation(obs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if ok:
            self.sent_observations += 1
        self.telemetry.write(
            "observation_sent",
            ok=ok,
            timestep=obs.get_timestep(),
            elapsed_ms=elapsed_ms,
            sent_observations=self.sent_observations,
        )
        return ok

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction], aggregate_fn=None):
        self.action_chunks_received += 1
        self.actions_received += len(incoming_actions)
        self.telemetry.write(
            "actions_received",
            chunk_size=len(incoming_actions),
            action_chunks_received=self.action_chunks_received,
            actions_received=self.actions_received,
            first_timestep=incoming_actions[0].get_timestep() if incoming_actions else None,
            last_timestep=incoming_actions[-1].get_timestep() if incoming_actions else None,
        )
        return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        start = time.perf_counter()
        performed = super().control_loop_action(verbose=verbose)
        self.actions_executed += 1
        self.telemetry.write(
            "action_executed",
            actions_executed=self.actions_executed,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            latest_action=self.latest_action,
            performed_action=performed,
        )
        return performed

    def stop(self):
        self.telemetry.write(
            "client_stop_requested",
            sent_observations=self.sent_observations,
            action_chunks_received=self.action_chunks_received,
            actions_received=self.actions_received,
            actions_executed=self.actions_executed,
        )
        try:
            return super().stop()
        finally:
            self.telemetry.write("client_stopped")


@draccus.wrap()
def async_client(cfg: PI0FastRobotClientConfig):
    logging.info("PI0-Fast async client config: %s", cfg)
    client = PI0FastRobotClient(cfg)

    if client.start():
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        timer = threading.Timer(cfg.duration_s, client.shutdown_event.set)
        timer.start()
        try:
            client.control_loop(task=cfg.task, verbose=True)
        finally:
            timer.cancel()
            client.stop()
            action_receiver_thread.join(timeout=max(2.0, cfg.duration_s))
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("PI0-Fast client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
