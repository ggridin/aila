param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$User,

    [Parameter(Mandatory = $true)]
    [string]$TargetPath,

    [string]$SourcePath = (Split-Path -Parent $PSScriptRoot),

    [int]$ConnectTimeoutSeconds = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
    throw "SourcePath does not exist: $SourcePath"
}

$Remote = "$User@$HostName"
$PathTrimChars = [char[]]@([char]"\", [char]"/")
$TargetTrimChars = [char[]]@([char]"/")
$NormalizedSource = (Resolve-Path -LiteralPath $SourcePath).Path.TrimEnd($PathTrimChars) + [System.IO.Path]::DirectorySeparatorChar
$NormalizedTarget = $TargetPath.TrimEnd($TargetTrimChars)
$Destination = "${Remote}:$NormalizedTarget/"

Invoke-Native -FilePath "ssh" -Arguments @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$ConnectTimeoutSeconds",
    $Remote,
    "mkdir -p '$NormalizedTarget'"
)

Invoke-Native -FilePath "rsync" -Arguments @(
    "-az",
    "--delete",
    "-e", "ssh -o BatchMode=yes -o ConnectTimeout=$ConnectTimeoutSeconds",
    "--exclude", ".git/",
    "--exclude", ".specrunner/",
    "--exclude", ".pytest_cache/",
    "--exclude", "__pycache__/",
    $NormalizedSource,
    $Destination
)
