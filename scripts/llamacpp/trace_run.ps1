# Start a thinking-trace run against the llama-server that serve.ps1 started (Windows side).
#   powershell -File scripts\llamacpp\trace_run.ps1 -Run tsmoke-1 [-Url http://127.0.0.1:8090] [-Workers 16]
# The run folder must exist (`ask-trainer traces prep --name <run>` inside WSL). The generator
# runs in the background with the standard-library Python; its pid goes to <run>\tracegen.pid
# (the dashboard feed reads it) and its output to <run>\tracegen.*.txt. A run started again
# skips the answers already in results.jsonl.
param([Parameter(Mandatory)][string]$Run, [string]$Url = 'http://127.0.0.1:8090', [int]$Workers = 16,
      [string]$Python = 'C:\Python314\python.exe')
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runDir = "F:\nfl-ask-trainer-data\ask\traces\$Run"
if (-not (Test-Path "$runDir\questions.jsonl")) { Write-Output "no questions in $runDir (run traces prep first)"; exit 1 }
$env:PYTHONPATH = "$repo\src"
$p = Start-Process -FilePath $Python -ArgumentList @('-m', 'asktrainer.tracegen', '--run', $runDir, '--url', $Url, '--workers', "$Workers") `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$runDir\tracegen.out.txt" -RedirectStandardError "$runDir\tracegen.err.txt"
$p.Id | Set-Content -Encoding ascii "$runDir\tracegen.pid"
Write-Output "tracegen started (pid $($p.Id)) for $runDir against $Url with $Workers workers"
