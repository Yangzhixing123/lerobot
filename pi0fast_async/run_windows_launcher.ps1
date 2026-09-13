param(
    [string]$ServerAddress = "58.199.161.50:8080",
    [string]$Policy = "yangzhixing/pi0fast-base-lora-so101-box-to-plate",
    [double]$DurationS = 5.0,
    [int]$Repeats = 1,
    [switch]$DisableTorqueOnDisconnect
)

$ErrorActionPreference = "Stop"
$Project = Resolve-Path (Join-Path $PSScriptRoot "..")
$Task = "Pick up the box and place it on the plate next to it."

function Write-Header {
    Clear-Host
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host " PI0-Fast SO101 Async Launcher" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host " Server   : $ServerAddress"
    Write-Host " Policy   : $Policy"
    Write-Host " Duration : $DurationS s"
    Write-Host " Repeats  : $Repeats"
    Write-Host " Task     : $Task"
    Write-Host " Project  : $Project"
    Write-Host ""
}

function Invoke-PI0FastRun {
    for ($i = 1; $i -le $Repeats; $i++) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
        Write-Host " Running repeat $i/$Repeats" -ForegroundColor Cyan
        Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray

        $RunArgs = @{
            ServerAddress = $ServerAddress
            Project = $Project
            Policy = $Policy
            Task = $Task
            DurationS = $DurationS
        }
        if ($DisableTorqueOnDisconnect) {
            $RunArgs["DisableTorqueOnDisconnect"] = $true
        }

        & "$Project\pi0fast_async\run_windows_client_once.ps1" @RunArgs

        if ($i -lt $Repeats) {
            Write-Host ""
            Write-Host "Repeat $i completed. Reset the scene, then press ENTER for the next repeat." -ForegroundColor Yellow
            Read-Host | Out-Null
        }
    }
}

:LauncherLoop while ($true) {
    Write-Header
    Write-Host " Select:" -ForegroundColor White
    Write-Host "   R  run"
    Write-Host "   S  set server address"
    Write-Host "   P  set policy repo/path"
    Write-Host "   T  set task text"
    Write-Host "   D  set duration seconds"
    Write-Host "   N  set repeats"
    Write-Host "   Q  quit"
    Write-Host ""

    $Choice = (Read-Host "Choice").Trim()

    switch -Regex ($Choice) {
        "^[Rr]$" {
            Invoke-PI0FastRun
            Write-Host ""
            Read-Host "Run complete. Press ENTER to return to launcher" | Out-Null
        }
        "^[Ss]$" {
            $Value = Read-Host "Server address [$ServerAddress]"
            if ($Value.Trim()) { $ServerAddress = $Value.Trim() }
        }
        "^[Pp]$" {
            $Value = Read-Host "Policy repo/path [$Policy]"
            if ($Value.Trim()) { $Policy = $Value.Trim() }
        }
        "^[Tt]$" {
            $Value = Read-Host "Task [$Task]"
            if ($Value.Trim()) { $Task = $Value.Trim() }
        }
        "^[Dd]$" {
            $Value = Read-Host "Duration seconds [$DurationS]"
            if ($Value.Trim()) { $DurationS = [double]$Value }
        }
        "^[Nn]$" {
            $Value = Read-Host "Repeats [$Repeats]"
            if ($Value.Trim()) {
                $Repeats = [int]$Value
                if ($Repeats -lt 1) { $Repeats = 1 }
            }
        }
        "^[Qq]$" {
            break LauncherLoop
        }
        default {
            Write-Host "Unknown choice: $Choice" -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
}
