import pickle
import base64

def restore_session(cookie_value: str):
    raw = base64.b64decode(cookie_value)
    return pickle.loads(raw)
