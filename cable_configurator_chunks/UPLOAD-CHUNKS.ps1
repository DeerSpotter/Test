param(
    [Parameter(Mandatory=$false)]
    [string]$ChunkFolder = "."
)

$ErrorActionPreference = "Stop"
$repo = "https://github.com/DeerSpotter/Test.git"
$work = Join-Path $env:TEMP "DeerSpotter-Test-chunk-upload"
$destRel = "cable_configurator_chunks"

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
git clone $repo $work
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

$parts = Get-ChildItem -LiteralPath $ChunkFolder -File | Where-Object {
    $_.Name -match '\.zip\.part00[1-4]$'
} | Sort-Object Name

if ($parts.Count -ne 4) {
    throw "Expected exactly four .zip.part001-.part004 files in: $ChunkFolder"
}

$dest = Join-Path $work $destRel
New-Item -ItemType Directory -Force -Path $dest | Out-Null

foreach ($part in $parts) {
    Copy-Item -LiteralPath $part.FullName -Destination (Join-Path $dest $part.Name) -Force
}

Push-Location $work
try {
    git add -- $destRel
    git commit -m "Add Cable Cartridge Configurator 2.0.0-dev chunks"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
}
finally {
    Pop-Location
}

Write-Host "Uploaded all four chunks to DeerSpotter/Test."
