# Run observathon sim in Docker. Loads OPENAI_API_KEY from .env automatically.
# Usage:
#   .\scripts\run-sim.ps1 -Phase public
#   .\scripts\run-sim.ps1 -Phase practice
#   .\scripts\run-sim.ps1 -Phase public -Questions harness/private_injection_test.json -Out injection_test_output.json

param(
    [ValidateSet("practice", "public", "private")]
    [string]$Phase = "public",
    [string]$Questions = "",
    [string]$Out = "run_output.json",
    [int]$Concurrency = 8
)

$Root = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "load-env.ps1") -EnvFile (Join-Path $Root ".env")

if (-not $env:OPENAI_API_KEY) {
    Write-Error "OPENAI_API_KEY is empty. Edit .env first."
    exit 1
}

# Windows onedir build (e.g. private) ships as a folder with an .exe inside.
# Run it natively; Docker (Linux) cannot execute a Windows binary.
$nativeExe = Join-Path $Root "bin\$Phase\observathon-sim\observathon-sim.exe"

if (Test-Path $nativeExe) {
    Push-Location $Root
    try {
        $simArgs = @(
            "--config", "solution/config.json",
            "--wrapper", "solution/wrapper.py",
            "--out", $Out
        )
        if ($Questions) {
            $simArgs += @("--questions", $Questions)
        } elseif ($Phase -ne "practice") {
            $simArgs += @("--concurrency", "$Concurrency")
        }
        & $nativeExe @simArgs
    } finally {
        Pop-Location
    }
    return
}

# Linux single-file binary (practice/public): run inside Docker.
$qMount = ($Root -replace '\\', '/') + ":/lab"
$bash = "cd /lab && chmod +x bin/$Phase/observathon-sim && ./bin/$Phase/observathon-sim --config solution/config.json --wrapper solution/wrapper.py --out $Out"

if ($Questions) {
    $bash += " --questions $Questions"
} elseif ($Phase -ne "practice") {
    $bash += " --concurrency $Concurrency"
}

docker run --rm `
    -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
    -v $qMount `
    python:3.12-slim `
    bash -c $bash
