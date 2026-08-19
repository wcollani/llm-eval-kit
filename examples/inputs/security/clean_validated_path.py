from pathlib import Path

UPLOAD_ROOT = Path("/srv/uploads").resolve()


def read_upload(filename: str) -> bytes:
    """Looks like the traversal case, but the resolved path is checked to be inside the
    root, so ../../etc/passwd resolves outside and is rejected."""
    candidate = (UPLOAD_ROOT / filename).resolve()
    if not candidate.is_relative_to(UPLOAD_ROOT):
        raise ValueError("path escapes the upload root")
    return candidate.read_bytes()
