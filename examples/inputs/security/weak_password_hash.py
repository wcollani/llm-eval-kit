import hashlib

def store_password(db, user_id: int, password: str) -> None:
    digest = hashlib.md5(password.encode()).hexdigest()
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (digest, user_id))

def verify_password(db, user_id: int, password: str) -> bool:
    row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    return row[0] == hashlib.md5(password.encode()).hexdigest()
