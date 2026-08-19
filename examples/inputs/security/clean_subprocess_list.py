import subprocess


def archive_report(report_name: str) -> int:
    """Looks like the command-injection case, but there is no shell and argv is a list,
    so a name containing ; or $() is passed through as a literal filename."""
    proc = subprocess.run(
        ["tar", "-czf", f"/var/backups/{report_name}.tar.gz", "/var/reports"],
        shell=False,
        check=False,
        capture_output=True,
    )
    return proc.returncode
