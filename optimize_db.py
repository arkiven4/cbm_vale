import sqlite3
import os

folder = "db"

for filename in os.listdir(folder):
    if filename.endswith(".db"):
        db_path = os.path.join(folder, filename)
        print(f"Processing {db_path}...")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # List all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]

        for table in tables:
            # Check if table has a timestamp column
            cur.execute(f"PRAGMA table_info({table});")
            columns = [c[1] for c in cur.fetchall()]
            if "timestamp" in columns:
                try:
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_timestamp ON {table}(timestamp);")
                    print(f"  Index created on {table}.timestamp")
                except sqlite3.OperationalError as e:
                    print(f"  Failed on {table}: {e}")
        conn.commit()
        conn.close()
