param(
    [ValidateSet('deepseek','commands')][string]$Backend = 'deepseek',
    [string]$Config = 'Local/deepseek.local.json',
    [ValidateRange(10,600)][int]$Seconds = 180,
    [double]$TargetX = 0.55,
    [double]$TargetY = 0.02,
    [switch]$Record
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv-sim/Scripts/python.exe'
$run = 'runs/contact-walk-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$arguments = @('-X','utf8','tools/run_dialogue.py','--profile','VD28','--backend',$Backend,'--config',$Config,
    '--seconds',$Seconds,'--target-x',$TargetX,'--target-y',$TargetY,'--panel','--realtime','--foot-view','--fps','30','--out',$run)
if ($Record) { $arguments += '--video' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw ('Inspect ' + $run + '/run.json') }
