UPLOAD_ROOT = "/srv/uploads/"

def read_upload(filename: str) -> bytes:
    with open(UPLOAD_ROOT + filename, "rb") as fh:
        return fh.read()
