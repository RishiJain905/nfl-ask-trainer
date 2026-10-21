# Start or stop llama-server with Spark-X2.6-4B (Q8_0) on the RX 7800 XT. See docs/llamacpp.md.
#   powershell -File scripts\llamacpp\serve.ps1                 # start (port 8090, 16 slots sharing one 64k-token KV pool)
#   powershell -File scripts\llamacpp\serve.ps1 -Stop           # stop the server this script started
# Only one GPU job at a time: never start it while a training run or another GPU job is live.
# -Bin picks the llama.cpp build (default: Rishi's TurboQuant fork + the Spark port; the clean
# upstream build is F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream\build-rdna3-gfx1101\bin);
# -Extra adds server flags.
param([switch]$Stop, [int]$Port = 8090, [int]$Parallel = 16, [int]$Ctx = 65536,
      [string]$Model = 'F:\LOCAL-Models\XHToken\Spark-X2.6-4B-GGUF\Spark-X2.6-4B-Q8_0.gguf',
      [string]$Bin = 'F:\Personal\TQ-Optimizer-Test\atomic-llama-cpp-turboquant-spark\build-rdna3-gfx1101\bin',
      [string]$Extra = '')  # e.g. -Extra "-ctk q8_0 -ctv turbo4"
$ErrorActionPreference = 'Stop'
$logs = 'D:\nfl-ml-data\ask\llamacpp'
$pidFile = "$logs\server-$Port.pid"
New-Item -ItemType Directory -Force $logs | Out-Null

if ($Stop) {
    if (-not (Test-Path $pidFile)) { Write-Output "no pid file for port $Port"; exit 0 }
    $id = [int](Get-Content $pidFile)
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -Id $id -Confirm:$false; Write-Output "stopped llama-server pid $id" }
    else { Write-Output "pid $id is not a running llama-server; nothing stopped" }
    Remove-Item $pidFile -Confirm:$false
    exit 0
}

$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH   # hipBLAS / rocBLAS DLLs
$env:HIP_VISIBLE_DEVICES = '1'                                     # the discrete card (device 0 is the integrated GPU)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$serverArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '-ngl', '99', '-fa', 'on',
          '-c', "$Ctx", '-np', "$Parallel", '-kvu', '-t', '8', '--no-webui',
          # no host-RAM prompt cache: saving a finished request's state (~250 MB) stalls every slot
          # for 1.5-3 s; checkpoints only serve that cache (measured 2026-10-21, docs/llamacpp.md)
          '--cache-ram', '0', '-ctxcp', '0') + $Extra.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
$p = Start-Process -FilePath "$Bin\llama-server.exe" -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logs\server-$stamp.out.txt" -RedirectStandardError "$logs\server-$stamp.err.txt"
$p.Id | Set-Content $pidFile
for ($i = 0; $i -lt 120; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 2).StatusCode -eq 200) {
        Write-Output "llama-server ready on http://127.0.0.1:$Port (pid $($p.Id), log $logs\server-$stamp.err.txt)"; exit 0 } } catch {}
    if ($p.HasExited) { Write-Output "llama-server exited early (code $($p.ExitCode)); see $logs\server-$stamp.err.txt"; exit 1 }
    Start-Sleep -Seconds 1
}
Write-Output "llama-server not ready after 120 s; see $logs\server-$stamp.err.txt"; exit 1
