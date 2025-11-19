import sqlite3
from contextlib import contextmanager

@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    yield conn
    conn.commit()
    conn.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  keyword TEXT NOT NULL,
  sub TEXT,
  title TEXT,
  url TEXT UNIQUE,
  created_utc REAL,
  n_comments INTEGER DEFAULT 0,
  match_id INTEGER,
  FOREIGN KEY(match_id) REFERENCES matches(id)
);
CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_url TEXT NOT NULL,
  keyword TEXT NOT NULL,
  sub TEXT,
  author TEXT,
  body TEXT,
  created_utc REAL,
  sentiment REAL,
  emb_dim INTEGER,
  emb_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_comments_thread ON comments(thread_url);
CREATE INDEX IF NOT EXISTS idx_comments_keyword ON comments(keyword);

CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  season TEXT,
  date TEXT,
  home TEXT,
  away TEXT,
  result TEXT,
  odds_h REAL,
  odds_d REAL,
  odds_a REAL,
  p_h REAL,
  p_d REAL,
  p_a REAL,
  UNIQUE(season, date, home, away)
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
"""

def init_schema(db_path: str):
    with connect(db_path) as conn:
        cur = conn.cursor()
        for stmt in SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                try:
                    cur.execute(s + ";")
                except Exception:
                    pass
