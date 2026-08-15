# AILA deployment wrappers

These PowerShell wrappers prepare and inspect the dedicated target over SSH. They
accept `-HostName` and `-User` and rely on the operator's existing SSH agent or
local SSH configuration; no password, token, or key material is stored here.

Examples:

```powershell
.\deploy\check-target.ps1 -HostName aila-laptop.local -User aila
.\deploy\sync-to-target.ps1 -HostName aila-laptop.local -User aila -TargetPath /home/aila/AI-living-on-laptop
.\deploy\remote-lint.ps1 -HostName aila-laptop.local -User aila -ScriptPath .\scripts\install-prereqs.sh
.\deploy\run-remote.ps1 -HostName aila-laptop.local -User aila -Command 'bash -n /home/aila/AI-living-on-laptop/scripts/setup-hermes.sh'
.\deploy\verify-target.ps1 -HostName aila-laptop.local -User aila -TargetPath /home/aila/AI-living-on-laptop
```

`sync-to-target.ps1` requires `rsync` and OpenSSH client tools on the local
machine. The other wrappers require OpenSSH client tools.
