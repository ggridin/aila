param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$User,

    [Parameter(Mandatory = $true)]
    [Alias('File')]
    [string]$ScriptPath,

    [string]$RemoteTempDir = "/tmp/aila-deploy-lint",

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

if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    throw "ScriptPath does not exist: $ScriptPath"
}

$Remote = "$User@$HostName"
$LeafName = Split-Path -Leaf $ScriptPath
$RemotePath = "$RemoteTempDir/$LeafName"

Invoke-Native -FilePath "ssh" -Arguments @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$ConnectTimeoutSeconds",
    $Remote,
    "mkdir -p '$RemoteTempDir'"
)

Invoke-Native -FilePath "scp" -Arguments @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$ConnectTimeoutSeconds",
    $ScriptPath,
    "${Remote}:$RemotePath"
)

Invoke-Native -FilePath "ssh" -Arguments @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$ConnectTimeoutSeconds",
    $Remote,
    "bash -n '$RemotePath'"
)
