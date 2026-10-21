# Quantise Spark-X2.6-4B to Q8 variants and measure them against BF16 (fidelity, then speed).
# Results of the 2026-10-20 run: docs/llamacpp.md. Steps, in order:
#   powershell -File scripts\llamacpp\quant_eval.ps1 -Step quantize|kldbase|kld|bench
#   powershell -File scripts\llamacpp\quant_eval.ps1 -Step batched -Variant Q8_0
# -Dir holds the GGUFs (the BF16 one is archived on D: after a run; copy it back or pass -Dir).
param([Parameter(Mandatory)][string]$Step, [string]$Variant = '', [string]$Dir = 'F:\LOCAL-Models\XHToken\Spark-X2.6-4B-GGUF')
$ErrorActionPreference = 'Stop'
$bin = 'F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream\build-rdna3-gfx1101\bin'
$gg = $Dir
$out = 'D:\nfl-ml-data\ask\quant\spark-x2.6-4b'
$text = "$out\kld-domain.txt"
$base = "$gg\kld-base-bf16.bin"   # temporary BF16 logits (~6 GB), deleted after the comparison
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:HIP_VISIBLE_DEVICES = '1'      # the discrete RX 7800 XT (device 0 is the integrated GPU)
New-Item -ItemType Directory -Force $out | Out-Null
$variants = [ordered]@{
    'Q8_0'        = @()
    'Q8_0-embF16' = @('--token-embedding-type', 'f16')
}
function Run($name, $exe, [string[]]$a) {
    Write-Output "== $name  $(Get-Date -Format o)"
    $quoted = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath "$bin\$exe" -ArgumentList $quoted -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$out\$name.out.txt" -RedirectStandardError "$out\$name.err.txt"
    Write-Output "exit $($p.ExitCode)  $(Get-Date -Format o)"
}
switch ($Step) {
    'quantize' {
        foreach ($v in $variants.GetEnumerator()) {
            Run "quantize-$($v.Key)" 'llama-quantize.exe' ($v.Value + @("$gg\Spark-X2.6-4B-BF16.gguf", "$gg\Spark-X2.6-4B-$($v.Key).gguf", 'Q8_0'))
        }
    }
    'kldbase' {
        Run 'kld-base-BF16' 'llama-perplexity.exe' @('-m', "$gg\Spark-X2.6-4B-BF16.gguf", '-f', $text, '-c', '2048', '--chunks', '24', '-ngl', '99', '-fa', 'on', '--kl-divergence-base', $base)
    }
    'kld' {
        foreach ($v in $variants.Keys) {
            Run "kld-$v" 'llama-perplexity.exe' @('-m', "$gg\Spark-X2.6-4B-$v.gguf", '-c', '2048', '--chunks', '24', '-ngl', '99', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
        }
    }
    'bench' {
        foreach ($v in @('BF16') + @($variants.Keys)) {
            Run "bench-$v" 'llama-bench.exe' @('-m', "$gg\Spark-X2.6-4B-$v.gguf", '-ngl', '99', '-fa', '1', '-t', '8', '-p', '512,2048', '-n', '128', '-r', '3', '-o', 'json')
        }
    }
    'batched' {
        Run "batched-$Variant" 'llama-batched-bench.exe' @('-m', "$gg\Spark-X2.6-4B-$Variant.gguf", '-ngl', '99', '-fa', 'on', '-c', '65536', '-b', '2048', '-ub', '512', '-npp', '1024', '-ntg', '512', '-npl', '1,4,8,16,32', '--output-format', 'jsonl')
    }
}
