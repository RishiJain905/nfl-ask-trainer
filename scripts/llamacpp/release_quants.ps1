# The release GGUFs of the merged SFT model (docs/release.md): convert, quantise, measure. Steps:
#   powershell -File scripts\llamacpp\release_quants.ps1 -Step convert|calib|imatrix|quantize|kldbase|kld|bench
# convert: the merged bf16 model to a BF16 GGUF (upstream's converter, so mainline llama.cpp loads
# it); calib + imatrix: an importance matrix from training text (make_calib_text.py) for the 4- and
# 6-bit quants; quantize: the variants below into -Dir, the folder that is uploaded; kldbase + kld:
# each variant against BF16 on the held-out validation text (make_kld_text.py's); bench: speed.
param([Parameter(Mandatory)][string]$Step,
      [string]$Name = 'Spark-X2.6-4B-SQL-Cypher-Agent',
      [string]$Merged = '',
      [string]$Dir = '')
$ErrorActionPreference = 'Stop'
if (-not $Merged) { $Merged = "D:\nfl-ml-data\ask\release\$Name" }
if (-not $Dir) { $Dir = "D:\nfl-ml-data\ask\release\$Name-GGUF" }
$bin = 'F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream\build-rdna3-gfx1101\bin'
$upstream = '/mnt/f/Personal/TQ-Optimizer-Test/llama-cpp-upstream'
$work = "D:\nfl-ml-data\ask\quant\$($Name.ToLower())"
$bf16 = "$work\$Name-BF16.gguf"                       # the reference; not uploaded
$calib = "$work\calib.txt"
$imatrix = "$work\imatrix.gguf"
$text = 'D:\nfl-ml-data\ask\quant\spark-x2.6-4b\kld-domain.txt'   # 240 held-out validation examples
$base = "$work\kld-base-bf16.bin"                     # temporary BF16 logits (~6 GB), deleted after
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:HIP_VISIBLE_DEVICES = '1'      # the discrete RX 7800 XT (device 0 is the integrated GPU)
New-Item -ItemType Directory -Force $work, $Dir | Out-Null
# XL = the K-quant with the tied embedding / output head at Q8_0 and every attention matrix one
# step up (Q6_K -> Q8_0, Q4_K_M -> Q6_K); the feed-forward bulk (69% of the weights) stays put
$variants = [ordered]@{
    'Q8_0'    = @{ type = 'Q8_0';   args = @() }
    'Q6_K'    = @{ type = 'Q6_K';   args = @('--imatrix', $imatrix) }
    'Q6_K_XL' = @{ type = 'Q6_K';   args = @('--imatrix', $imatrix, '--token-embedding-type', 'q8_0',
                                             '--tensor-type', 'attn_qkv=q8_0', '--tensor-type', 'attn_output=q8_0') }
    'Q4_K_M'  = @{ type = 'Q4_K_M'; args = @('--imatrix', $imatrix) }
    'Q4_K_XL' = @{ type = 'Q4_K_M'; args = @('--imatrix', $imatrix, '--token-embedding-type', 'q8_0',
                                             '--tensor-type', 'attn_qkv=q6_k', '--tensor-type', 'attn_output=q6_k') }
}
function Run($label, $exe, [string[]]$a) {
    Write-Output "== $label  $(Get-Date -Format o)"
    $quoted = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath "$bin\$exe" -ArgumentList $quoted -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$work\$label.out.txt" -RedirectStandardError "$work\$label.err.txt"
    Write-Output "exit $($p.ExitCode)  $(Get-Date -Format o)"
    if ($p.ExitCode -ne 0) { throw "$label failed (exit $($p.ExitCode)): see $work\$label.err.txt" }
}
function InWsl($cmd) {
    wsl.exe -d Ubuntu -- bash -lc "source ~/venvs/nfl-ask-trainer/bin/activate && $cmd"
    if ($LASTEXITCODE -ne 0) { throw "WSL step failed: $cmd" }
}
function WslPath($p) { '/mnt/' + $p.Substring(0, 1).ToLower() + $p.Substring(2).Replace('\', '/') }
switch ($Step) {
    'convert' {
        InWsl "cd $upstream && PYTHONPATH=gguf-py python convert_hf_to_gguf.py $(WslPath $Merged) --outtype bf16 --outfile $(WslPath $bf16)"
    }
    'calib' {
        InWsl "cd /mnt/f/Personal/NFL/nfl-ask-trainer && python scripts/llamacpp/make_calib_text.py $(WslPath $Merged) /mnt/d/nfl-ml-data/ask/training/public $(WslPath $calib) 40"
    }
    'imatrix' {
        Run 'imatrix' 'llama-imatrix.exe' @('-m', $bf16, '-f', $calib, '-o', $imatrix, '-c', '2048', '-ngl', '99', '-fa', 'on')
    }
    'quantize' {
        foreach ($v in $variants.GetEnumerator()) {
            Run "quantize-$($v.Key)" 'llama-quantize.exe' ($v.Value.args + @($bf16, "$Dir\$Name-$($v.Key).gguf", $v.Value.type))
        }
    }
    'kldbase' {
        Run 'kld-base-BF16' 'llama-perplexity.exe' @('-m', $bf16, '-f', $text, '-c', '2048', '--chunks', '24', '-ngl', '99', '-fa', 'on', '--kl-divergence-base', $base)
    }
    'kld' {
        foreach ($v in $variants.Keys) {
            Run "kld-$v" 'llama-perplexity.exe' @('-m', "$Dir\$Name-$v.gguf", '-c', '2048', '--chunks', '24', '-ngl', '99', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
        }
        Remove-Item $base -Confirm:$false
    }
    'bench' {
        foreach ($v in @('BF16') + @($variants.Keys)) {
            $m = if ($v -eq 'BF16') { $bf16 } else { "$Dir\$Name-$v.gguf" }
            Run "bench-$v" 'llama-bench.exe' @('-m', $m, '-ngl', '99', '-fa', '1', '-t', '8', '-p', '512,2048', '-n', '128', '-r', '3', '-o', 'json')
        }
    }
}
