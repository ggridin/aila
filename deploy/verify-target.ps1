param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$User,

    [Parameter(Mandatory = $true)]
    [string]$TargetPath,

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

function ConvertTo-RemoteSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $Escaped = $Value.Replace("'", "'\''")
    return "'$Escaped'"
}

$Remote = "$User@$HostName"
$QuotedTarget = ConvertTo-RemoteSingleQuoted -Value $TargetPath
$RemoteCommand = @(
    "set -e",
    'export PATH="$HOME/.local/bin:$HOME/.hermes/bin:$PATH"',
    "echo '== host =='",
    "uname -a",
    "echo '== target path =='",
    "test -d $QuotedTarget",
    "printf '%s\n' $QuotedTarget",
    "echo '== repository files =='",
    "find $QuotedTarget -maxdepth 2 -type f | sort | sed -n '1,80p'",
    "echo '== python =='",
    "command -v python3 || true",
    "python3 --version || true",
    "echo '== bash =='",
    "bash --version | sed -n '1p'",
    "echo '== hermes =='",
    "command -v hermes || true",
    "hermes --version || true",
    "echo '== aila user units =='",
    "systemctl --user list-units 'aila-*' --no-pager --plain || true"
) -join "`n"

Invoke-Native -FilePath "ssh" -Arguments @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$ConnectTimeoutSeconds",
    $Remote,
    $RemoteCommand
)
