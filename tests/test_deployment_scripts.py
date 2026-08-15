from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "deploy"

SCRIPTS = {
    "check-target.ps1",
    "sync-to-target.ps1",
    "remote-lint.ps1",
    "run-remote.ps1",
    "verify-target.ps1",
}


def _script(name: str) -> str:
    return (DEPLOY_ROOT / name).read_text(encoding="utf-8")


def _param_names(text: str) -> set[str]:
    param_block = re.search(r"param\((.*?)\)\s*Set-StrictMode", text, re.DOTALL)
    assert param_block is not None
    return {
        string_name or int_name
        for string_name, int_name in re.findall(
            r"\[string\]\$(\w+)|\[int\]\$(\w+)", param_block.group(1)
        )
    }


def _mandatory_string_param(text: str, name: str) -> bool:
    return (
        re.search(
            rf"\[Parameter\(Mandatory = \$true\)\]\s*(?:\[[^\]]+\]\s*)*\[string\]\${name}\b",
            text,
            re.DOTALL,
        )
        is not None
    )


def test_deploy_directory_contains_expected_wrappers_and_readme() -> None:
    files = {path.name for path in DEPLOY_ROOT.iterdir() if path.is_file()}

    assert files == SCRIPTS | {"README.md"}
    readme = (DEPLOY_ROOT / "README.md").read_text(encoding="utf-8")
    for script in SCRIPTS:
        assert script in readme


def test_all_wrappers_require_target_identity_without_secret_parameters() -> None:
    forbidden = re.compile(r"\$(Password|Credential|Token|ApiKey|Secret)\b", re.IGNORECASE)

    for script in SCRIPTS:
        text = _script(script)
        assert _mandatory_string_param(text, "HostName")
        assert _mandatory_string_param(text, "User")
        assert '"$User@$HostName"' in text
        assert "BatchMode=yes" in text
        assert forbidden.search(text) is None


def test_check_target_only_tests_ssh_reachability() -> None:
    text = _script("check-target.ps1")

    assert _param_names(text) == {"HostName", "User", "ConnectTimeoutSeconds"}
    assert 'Invoke-Native -FilePath "ssh"' in text
    assert "AILA_TARGET_OK" in text
    assert "scp" not in text
    assert "rsync" not in text


def test_sync_to_target_mirrors_repo_with_rsync_delete_and_excludes_local_state() -> None:
    text = _script("sync-to-target.ps1")

    assert {"HostName", "User", "TargetPath", "SourcePath", "ConnectTimeoutSeconds"} == _param_names(
        text
    )
    assert _mandatory_string_param(text, "TargetPath")
    assert "Split-Path -Parent $PSScriptRoot" in text
    assert 'Invoke-Native -FilePath "rsync"' in text
    assert '"--delete"' in text
    assert '".git/"' in text
    assert '".specrunner/"' in text
    assert '"${Remote}:$NormalizedTarget/"' in text


def test_remote_lint_copies_one_script_and_runs_bash_syntax_check() -> None:
    text = _script("remote-lint.ps1")

    assert _mandatory_string_param(text, "ScriptPath")
    assert "/tmp/aila-deploy-lint" in text
    assert "Split-Path -Leaf $ScriptPath" in text
    assert 'Invoke-Native -FilePath "scp"' in text
    assert '"bash -n ' in text


def test_run_remote_executes_operator_supplied_command_over_ssh() -> None:
    text = _script("run-remote.ps1")

    assert _param_names(text) == {"HostName", "User", "Command", "ConnectTimeoutSeconds"}
    assert _mandatory_string_param(text, "Command")
    assert 'Invoke-Native -FilePath "ssh"' in text
    assert "$Command" in text
    assert "scp" not in text
    assert "rsync" not in text


def test_verify_target_collects_read_only_smoke_evidence() -> None:
    text = _script("verify-target.ps1")

    assert _mandatory_string_param(text, "TargetPath")
    for expected in (
        "uname -a",
        "test -d $QuotedTarget",
        "find $QuotedTarget -maxdepth 2 -type f",
        "python3 --version",
        "bash --version",
        "hermes --version",
        "systemctl --user list-units 'aila-*'",
    ):
        assert expected in text
    for forbidden in ("sudo", "apt ", "install-prereqs", "setup-hermes"):
        assert forbidden not in text
