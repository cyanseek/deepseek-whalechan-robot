param(
    [ValidateSet('deepseek','commands')][string]$Backend = 'commands',
    [string]$Config = 'Local/deepseek.local.json',
    [ValidateRange(10,600)][int]$Seconds = 180,
    [switch]$Record
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$behaviorPython = Join-Path $PSScriptRoot '.venv-sim/Scripts/python.exe'
$behaviorRun = 'runs/behaviors-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$behaviorArgs = @('-X','utf8','tools/run_dialogue.py','--profile','VD29','--behaviors',
    '--backend',$Backend,'--config',$Config,'--seconds',$Seconds,'--target-x','0.55','--target-y','0.02',
    '--panel','--realtime','--foot-view','--fps','30','--out',$behaviorRun)
if ($Record) { $behaviorArgs += '--video' }
& $behaviorPython @behaviorArgs
if ($LASTEXITCODE -ne 0) { throw ('Inspect ' + $behaviorRun + '/run.json') }
