param(
    [ValidateSet('interactive','stand','greet','walk','approach','cancel','loss','switch','look','push','jump','turn','side','expressions','showcase','tail','tail-cancel','coordinated','coord-cancel')]
    [string]$Scenario = 'interactive',
    [ValidateRange(1,180)][int]$Seconds = 180,
    [switch]$Record,
    [switch]$CollisionView,
    [switch]$Presentation,
    [ValidateSet('VD24','VD23','VD22','VD21','VD2','VD1')][string]$Profile = 'VD24',
    [string]$Delivery,
    [ValidateSet(30,60)][int]$Fps = 30
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv-sim/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing .venv-sim Python was not found.' }
$output = 'runs/' + $Profile + '-live-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$entry = if ($Profile -eq 'VD1') { 'tools/run_bluefatfish.py' } else { 'tools/run_bluefatfish_v2.py' }
$arguments = @('-X', 'utf8', $entry, '--out', $output,
    '--scenario', $Scenario, '--seconds', $Seconds, '--live')
if ($Profile -ne 'VD1') { $arguments += @('--profile', $Profile) }
if ($Delivery) { $arguments += @('--delivery', $Delivery) }
if ($Profile -ne 'VD1') { $arguments += @('--fps', $Fps) }
if ($Record) { $arguments += '--video' }
if ($CollisionView) { $arguments += '--collision-view' }
if ($Presentation -and $Profile -ne 'VD1') { $arguments += '--presentation' }
Write-Host 'OpenDafeiyu: W forward / A turn / S back / J small jump / E expressions / T tail.'
Write-Host 'R approach red / G greet / C or Space cancel / V camera / D collision view.'
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw ('Simulation did not pass; inspect ' + $output) }
Write-Host ('Evidence saved in ' + $output)
