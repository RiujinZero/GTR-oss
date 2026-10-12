param(
    [Parameter(Mandatory = $true)]
    [int]$GatePid,
    [double]$GatePpl = 43.72,
    [int]$MinimumGateStep = 6000,
    [double]$RuntimeHours = 10.0,
    [double]$ReportMinutes = 15.0,
    [string]$PythonExecutable = "C:\Users\user\Documents\Qwentin iOS Beta\hedgehogAI\Tools\qwen_cpu_env\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$gateName = "convergent_language_130k_r2_sw50_seed0_10k"
$runName = "convergent_language_130k_r2_sw50_seed0_10h"
$gateLog = Join-Path $repo "experimental_checkpoints\$gateName.out.log"
$gateCheckpoint = Join-Path $repo "experimental_checkpoints\$gateName\best.pt"
$runDirectory = Join-Path $repo "experimental_checkpoints\$runName"
$runLog = Join-Path $repo "experimental_checkpoints\$runName.out.log"
$runError = Join-Path $repo "experimental_checkpoints\$runName.err.log"
$watchLog = Join-Path $repo "experimental_checkpoints\$runName.launch.log"

$gateProcess = Get-Process -Id $GatePid -ErrorAction SilentlyContinue
if ($null -ne $gateProcess) {
    Wait-Process -Id $GatePid
}

$stepLines = Select-String -Path $gateLog -Pattern "^step=[0-9]+ "
$lastStep = 0
foreach ($line in $stepLines) {
    if ($line.Line -match "^step=([0-9]+) ") {
        $lastStep = [Math]::Max($lastStep, [int]$Matches[1])
    }
}
if ($lastStep -lt $MinimumGateStep) {
    "launch_aborted reason=insufficient_gate_steps last_step=$lastStep minimum_step=$MinimumGateStep gate_log=$gateLog" |
        Set-Content -Path $watchLog
    exit 2
}
$observedPpl = @(
    foreach ($line in $stepLines) {
        if ($line.Line -match "ppl=([0-9.]+)") {
            [double]$Matches[1]
        }
    }
)
if (-not $observedPpl.Count) {
    "launch_aborted reason=missing_ppl gate_log=$gateLog" |
        Set-Content -Path $watchLog
    exit 3
}
$bestPpl = ($observedPpl | Measure-Object -Minimum).Minimum
if ($bestPpl -ge $GatePpl) {
    "launch_aborted reason=gate_failed best_ppl=$bestPpl threshold=$GatePpl" |
        Set-Content -Path $watchLog
    exit 4
}
if (-not (Test-Path -LiteralPath $gateCheckpoint -PathType Leaf)) {
    "launch_aborted reason=missing_checkpoint path=$gateCheckpoint" |
        Set-Content -Path $watchLog
    exit 5
}

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$runtimeSeconds = [int][Math]::Round($RuntimeHours * 3600)
$reportSeconds = [int][Math]::Round($ReportMinutes * 60)
$arguments = @(
    "-u",
    "gator_language_train.py",
    "--data", "C:\Users\user\Downloads\smoltalk_everyday.jsonl",
    "--lmsys-data", "C:\Users\user\Downloads\lmsys_chat_1m.jsonl",
    "--lmsys-weight", "1.0",
    "--streaming-weight", "1.0",
    "--streaming-exhaustion", "restart",
    "--primary-stream-weight", "1",
    "--primary-shuffle-buffer", "8192",
    "--lmsys-source-weight", "4",
    "--lmsys-load-mode", "stream",
    "--lmsys-shuffle-buffer", "16384",
    "--lmsys-validation-conversations", "512",
    "--lmsys-max-characters", "512",
    "--lmsys-min-ascii-letter-ratio", "0.9",
    "--lmsys-min-response-characters", "8",
    "--lmsys-require-user-prompt",
    "--chat-data", "C:\Users\user\Downloads\tulu3_sft.jsonl",
    "--chat-data", "C:\Users\user\Downloads\smoltalk_preference.jsonl",
    "--chat-source-weight", "3",
    "--chat-source-weight", "2",
    "--architecture", "convergent",
    "--steps", "200000",
    "--max-runtime-seconds", "$runtimeSeconds",
    "--batch-size", "32",
    "--eval-batch-size", "64",
    "--log-every", "200000",
    "--log-every-seconds", "$reportSeconds",
    "--logging-slim",
    "--vocab-size", "2048",
    "--context-tokens", "64",
    "--response-tokens", "32",
    "--embedding-dim", "32",
    "--d-model", "20",
    "--nodes", "66",
    "--graph-degree", "3",
    "--lanes", "10",
    "--relays", "2",
    "--connection-layout", "small_world",
    "--connection-seed", "0",
    "--connection-rewire", "0.5",
    "--skip-first-assistant",
    "--require-complete-context",
    "--trim-context-turns",
    "--deduplicate-examples",
    "--scheduler", "cosine",
    "--scheduler-total-steps", "300000",
    "--minimum-lr", "0.0001",
    "--learning-rate", "0.001",
    "--seed", "0",
    "--batch-pipeline", "serial",
    "--resume-checkpoint", $gateCheckpoint,
    "--resume-optimizer",
    "--resume-data-state", "reset",
    "--checkpoint-dir", $runDirectory
)

# Codex's Windows process environment can contain both ``PATH`` and ``Path``.
# PowerShell 5.1's Start-Process rejects that case-insensitive duplicate while
# constructing the child environment. Collapse the pair before launching.
$pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")

$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $runLog `
    -RedirectStandardError $runError `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -Path (Join-Path $runDirectory "process.pid")
@(
    "launch_started pid=$($process.Id)",
    "gate_last_step=$lastStep minimum_step=$MinimumGateStep",
    "gate_best_ppl=$bestPpl threshold=$GatePpl",
    "runtime_hours=$RuntimeHours report_minutes=$ReportMinutes",
    "python=$PythonExecutable",
    "checkpoint=$gateCheckpoint",
    "stdout=$runLog",
    "stderr=$runError"
) | Set-Content -Path $watchLog
