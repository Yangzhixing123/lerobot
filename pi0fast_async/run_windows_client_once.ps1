param(
    [string]$ServerAddress = "58.199.161.50:8080",
    [string]$Project = "",
    [string]$Policy = "yangzhixing/pi0fast-base-lora-so101-box-to-plate",
    [string]$RobotPort = "COM8",
    [string]$RobotId = "so101_pi0fast_follower",
    [string]$Task = "Pick up the box and place it on the plate next to it.",
    [double]$DurationS = 5.0,
    [int]$Fps = 30,
    [int]$ActionsPerChunk = 10,
    [double]$ChunkSizeThreshold = 0.5,
    [switch]$DisableTorqueOnDisconnect
)

$ErrorActionPreference = "Stop"

if (-not $Project) {
    $Project = Resolve-Path (Join-Path $PSScriptRoot "..")
}

$ClientLogDir = Join-Path $Project "pi0fast_async\logs\client"
New-Item -ItemType Directory -Force -Path $ClientLogDir | Out-Null

Set-Location $Project

Write-Host "PI0-Fast async client"
Write-Host "  server:              $ServerAddress"
Write-Host "  policy:              $Policy"
Write-Host "  task:                $Task"
Write-Host "  duration_s:          $DurationS"
Write-Host "  fps:                 $Fps"
Write-Host "  actions_per_chunk:   $ActionsPerChunk"
Write-Host "  robot_port:          $RobotPort"
Write-Host "  robot_id:            $RobotId"
Write-Host "  logs:                $ClientLogDir"

if ($DisableTorqueOnDisconnect) {
    $TorqueArg = "--robot.disable_torque_on_disconnect=true"
} else {
    $TorqueArg = "--robot.disable_torque_on_disconnect=false"
}

uv run python pi0fast_async\pi0fast_robot_client.py `
    --robot.type=so101_follower `
    --robot.port="$RobotPort" `
    --robot.id="$RobotId" `
    $TorqueArg `
    --robot.max_relative_target=5 `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}}" `
    --task="$Task" `
    --server_address="$ServerAddress" `
    --policy_type=pi0_fast `
    --pretrained_name_or_path="$Policy" `
    --policy_device=cuda `
    --client_device=cpu `
    --actions_per_chunk=$ActionsPerChunk `
    --chunk_size_threshold=$ChunkSizeThreshold `
    --aggregate_fn_name=weighted_average `
    --fps=$Fps `
    --duration_s=$DurationS `
    --rename_map="{}" `
    --log_dir="$ClientLogDir"
