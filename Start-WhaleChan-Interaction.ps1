param(
    [ValidateSet('deepseek','commands')][string]$Backend = 'deepseek',
    [string]$Config = 'Local/deepseek.local.json',
    [ValidateRange(10,600)][int]$Seconds = 180,
    [switch]$Record
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv-sim/Scripts/python.exe'
$run = 'runs/interaction-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$arguments = @('-X','utf8','tools/run_dialogue.py','--backend',$Backend,'--config',$Config,'--seconds',$Seconds,'--panel','--realtime','--out',$run)
if ($Record) { $arguments += '--video' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw ('Inspect ' + $run + '/run.json') }
