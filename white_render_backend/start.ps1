param(
  [ValidateSet('api','worker')][string]$Mode='api',
  [string]$Python='python',
  [string]$DataRoot='',
  [int]$Port=8018
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $DataRoot) { $DataRoot=Join-Path $PSScriptRoot 'work' }
$pipelineRoot=[System.IO.Path]::GetFullPath($DataRoot)
New-Item -ItemType Directory -Force -Path $pipelineRoot | Out-Null
$auth=Join-Path $pipelineRoot 'local-auth.private.json'
if (-not (Test-Path -LiteralPath $auth)) {
  $bytes=New-Object byte[] 32
  $rng=[System.Security.Cryptography.RandomNumberGenerator]::Create()
  $rng.GetBytes($bytes);$rng.Dispose()
  $token=[Convert]::ToBase64String($bytes)
  $map=@{};$map[$token]=@{project='pipeline';roles=@('viewer','editor')}
  $map | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $auth -Encoding UTF8
}
$env:WHITE_RENDER_ROOT=$pipelineRoot
$env:WHITE_RENDER_AUTH_JSON=Get-Content -Raw -LiteralPath $auth
$credential=Join-Path $PSScriptRoot 'private\minimaxapi.docx'
if (Test-Path -LiteralPath $credential) { $env:MINIMAX_API_KEY_FILE=$credential }
$pythonDir=Split-Path -Parent $Python
if ($pythonDir) { $env:PATH="$pythonDir;$pythonDir\Library\bin;$pythonDir\Scripts;"+$env:PATH }
Write-Host "Project pipeline. Private platform token: $auth"
if ($Mode -eq 'api') { & $Python -m white_render_backend.cli --root $pipelineRoot serve --port $Port }
else { & $Python -m white_render_backend.cli --root $pipelineRoot worker }
exit $LASTEXITCODE
