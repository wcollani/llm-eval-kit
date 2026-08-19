import sqlite3

def find_user(conn: sqlite3.Connection, username: str):
    cur = conn.cursor()
    query = "SELECT id, email FROM users WHERE username = '" + username + "'"
    cur.execute(query)
    return cur.fetchone()
