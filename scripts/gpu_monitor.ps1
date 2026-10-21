# GPU / system monitor for the live dashboard (Windows side; run while a training run is live).
#
#   powershell -NoProfile -File scripts\gpu_monitor.ps1 [-OutDir F:\nfl-ask-trainer-ckpt\_monitor] [-Seconds 15] [-Hours 30]
#
# Appends one JSON line per sample to <OutDir>\gpu-<yyyyMMdd>.jsonl (UTC timestamps):
#   vram_gb      dedicated GPU memory in use (the busiest adapter: the RX 7800 XT while training)
#   cpu          total CPU busy, percent
#   ram_free_gb  free system memory
#   gpu_util     null: Windows doesn't attribute WSL's ROCm work to any GPU engine (every
#                engine reads ~0% while the GPU trains), so there is no honest utilisation number
# Reads Windows performance counters only. Stops after -Hours, or when the process is ended.
param(
  [string]$OutDir = "F:\nfl-ask-trainer-ckpt\_monitor",
  [int]$Seconds = 15,
  [double]$Hours = 30
)
New-Item -ItemType Directory -Force $OutDir | Out-Null
$end = (Get-Date).AddHours($Hours)
$counters = @('\GPU Adapter Memory(*)\Dedicated Usage', '\Processor(_Total)\% Processor Time', '\Memory\Available MBytes')
while ((Get-Date) -lt $end) {
  try {
    $s = (Get-Counter -Counter $counters -ErrorAction Stop).CounterSamples
    $vram = ($s | Where-Object { $_.Path -like '*gpu adapter memory*' } | Measure-Object CookedValue -Maximum).Maximum
    $cpu = ($s | Where-Object { $_.Path -like '*processor(_total)*' } | Select-Object -First 1).CookedValue
    $ram = ($s | Where-Object { $_.Path -like '*available mbytes*' } | Select-Object -First 1).CookedValue
    $row = [ordered]@{
      ts          = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss+00:00')
      vram_gb     = [math]::Round($vram / 1GB, 2)
      cpu         = [math]::Round($cpu, 1)
      ram_free_gb = [math]::Round($ram / 1024, 2)
      gpu_util    = $null
    }
    $file = Join-Path $OutDir ("gpu-{0}.jsonl" -f (Get-Date).ToUniversalTime().ToString('yyyyMMdd'))
    # UTF-8 without a BOM (PowerShell 5.1's Add-Content -Encoding utf8 writes one)
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($file, ($row | ConvertTo-Json -Compress) + "`n", $utf8)
  } catch {
    # a counter read can fail transiently; the next sample tries again
  }
  Start-Sleep -Seconds $Seconds
}
