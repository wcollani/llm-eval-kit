import sqlite3


def find_user(conn: sqlite3.Connection, username: str):
    """Looks like the injection case, but the value is bound, not concatenated."""
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE username = ?", (username,))
    return cur.fetchone()


def search_users(conn: sqlite3.Connection, fields: list[str], term: str):
    # The column list is interpolated, but only from a fixed allowlist -- never from input.
    allowed = {"username", "email", "display_name"}
    chosen = [f for f in fields if f in allowed] or ["username"]
    sql = f"SELECT {', '.join(chosen)} FROM users WHERE email LIKE ?"
    return conn.execute(sql, (f"%{term}%",)).fetchall()
