param(
    [ValidateSet('baseline','human','search','retrieval','adaptive','ablation','human_ablation','freeze','final','export')]
    [string]$Stage = 'baseline'
)
$ErrorActionPreference = 'Stop'
# Local convenience launcher for the environment prepared in this Codex workspace.
# Portable/Colab setup is documented separately in README.md.
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path $taskRoot 'work\trackc-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Prepared local environment not found. Follow README.md to install dependencies.'
}
$env:HF_HOME = Join-Path $taskRoot 'work\hf-cache'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
# This machine's Windows proxy uses HTTP CONNECT. Python 3.10.0 mislabels it HTTPS.
# Restrict this compatibility adjustment to this process and the observed local proxy.
if (-not $env:HTTPS_PROXY) {
    $proxySettings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
    if ($proxySettings.ProxyEnable -and $proxySettings.ProxyServer -eq '127.0.0.1:7890') {
        $env:HTTPS_PROXY = 'http://127.0.0.1:7890'
        $env:HTTP_PROXY = 'http://127.0.0.1:7890'
    }
}
$entry = if ($Stage -eq 'human_ablation') { 'ablate_human.py' } else { 'run_trackc.py' }
$taskArgs = @((Join-Path $PSScriptRoot $entry))
if ($Stage -ne 'human_ablation') { $taskArgs += $Stage }
& $taskPython @taskArgs
exit $LASTEXITCODE
