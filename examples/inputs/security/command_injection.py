import os

def archive_report(report_name: str) -> int:
    return os.system("tar -czf /var/backups/" + report_name + ".tar.gz /var/reports")
