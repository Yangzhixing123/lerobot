# PI0-Fast Async Server/Client

This directory is isolated from `src/lerobot`. It serves the PI0-Fast LoRA
adapter `yangzhixing/pi0fast-base-lora-so101-box-to-plate` from Linux and runs
the SO101 robot client on Windows through the upstream LeRobot async gRPC
protocol.

The Linux server loads the adapter as PEFT:

1. Read the adapter config.
2. Load `lerobot/pi0fast-base` with the adapter policy config.
3. Attach the LoRA adapter.
4. Load the adapter preprocessor/postprocessor.

## Linux Server

```bash
cd /data/ywb-2/yzx/lerobot
bash pi0fast_async/run_linux_server.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8080 bash pi0fast_async/run_linux_server.sh
```

The server uses:

```text
HF_HOME=/data/ywb-2/yzx/lerobot/.cache/huggingface
HF_HUB_CACHE=/data/ywb-2/yzx/lerobot/.cache/huggingface/hub
```

## Windows Client

From a Windows clone/copy of this repo:

```powershell
.\pi0fast_async\run_windows_client_once.ps1 -ServerAddress "SERVER_IP:8080" -DurationS 5
```

If your SO101 is on a different serial port:

```powershell
.\pi0fast_async\run_windows_client_once.ps1 -ServerAddress "SERVER_IP:8080" -RobotPort COM8
```

Interactive launcher:

```powershell
.\pi0fast_async\run_windows_launcher.ps1 -ServerAddress "SERVER_IP:8080"
```

Defaults match the downloaded adapter:

```text
policy_type=pi0_fast
pretrained_name_or_path=yangzhixing/pi0fast-base-lora-so101-box-to-plate
actions_per_chunk=10
task=Pick up the box and place it on the plate next to it.
cameras=front,wrist
```

This policy config expects:

```text
observation.state: 6 dims
observation.images.front
observation.images.wrist
action: 6 dims
```
