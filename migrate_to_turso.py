#!/usr/bin/env python3
"""
Script to migrate existing local SQLite data (arabic_study_history.db) to Turso libSQL cloud DB.
"""

import os
import sys
import sqlite3

try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        tomllib = None

LOCAL_DB_FILE = "arabic_study_history.db"

def get_turso_credentials():
    """Retrieve Turso URL and Auth Token from environment variables or .streamlit/secrets.toml."""
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")

    if not url or not token:
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            try:
                if tomllib:
                    with open(secrets_path, "rb") if hasattr(tomllib, "load") and tomllib.__name__ == "tomllib" else open(secrets_path, "r") as f:
                        secrets = tomllib.load(f)
                    url = url or secrets.get("TURSO_DATABASE_URL")
                    token = token or secrets.get("TURSO_AUTH_TOKEN")
            except Exception as e:
                print(f"Warning: Could not parse {secrets_path}: {e}")

    return url, token

def main():
    print("=" * 60)
    print("  Arabic Study Logs Migration Tool: Local SQLite -> Turso Cloud")
    print("=" * 60)

    # 1. Check local SQLite DB
    if not os.path.exists(LOCAL_DB_FILE):
        print(f"[-] Local database file '{LOCAL_DB_FILE}' not found. Nothing to migrate.")
        return

    local_conn = sqlite3.connect(LOCAL_DB_FILE)
    local_cursor = local_conn.cursor()

    # Verify table exists
    local_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='study_logs';")
    if not local_cursor.fetchone():
        print(f"[-] Table 'study_logs' not found in '{LOCAL_DB_FILE}'. Nothing to migrate.")
        local_conn.close()
        return

    local_cursor.execute("""
        SELECT id, timestamp, source_filename, image_base64, tashkeel_text, full_translation, verbs_json, nouns_json, particles_json, deep_sarf_json
        FROM study_logs
        ORDER BY id ASC
    """)
    local_rows = local_cursor.fetchall()
    print(f"[+] Found {len(local_rows)} study log records in local SQLite DB.")

    if not local_rows:
        print("[!] No records to migrate.")
        local_conn.close()
        return

    # 2. Get Turso credentials
    turso_url, turso_token = get_turso_credentials()
    if not turso_url or not turso_token:
        print("[-] Error: Turso credentials missing.")
        print("    Please set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN environment variables")
        print("    or add them to .streamlit/secrets.toml.")
        local_conn.close()
        sys.exit(1)

    print(f"[+] Connecting to Turso database at: {turso_url}")

    use_libsql_exp = False
    use_libsql_client = False

    try:
        import libsql_experimental as libsql
        use_libsql_exp = True
    except ImportError:
        try:
            import libsql_client
            use_libsql_client = True
        except ImportError:
            print("[-] Error: Neither 'libsql-experimental' nor 'libsql-client' package is installed.")
            print("    Please run: pip install libsql-experimental (or pip install libsql-client)")
            local_conn.close()
            sys.exit(1)

    try:
        migrated_count = 0
        skipped_count = 0
        total_turso_count = 0

        if use_libsql_exp:
            turso_conn = libsql.connect(database=turso_url, auth_token=turso_token)
            turso_cursor = turso_conn.cursor()

            turso_cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source_filename TEXT,
                    image_base64 TEXT,
                    tashkeel_text TEXT,
                    full_translation TEXT,
                    verbs_json TEXT,
                    nouns_json TEXT,
                    particles_json TEXT,
                    deep_sarf_json TEXT
                )
            """)
            turso_conn.commit()

            for row in local_rows:
                try:
                    turso_cursor.execute("""
                        INSERT OR IGNORE INTO study_logs 
                        (id, timestamp, source_filename, image_base64, tashkeel_text, full_translation, verbs_json, nouns_json, particles_json, deep_sarf_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, row)
                    if turso_cursor.rowcount > 0:
                        migrated_count += 1
                    else:
                        skipped_count += 1
                except Exception as row_err:
                    print(f"[!] Error inserting record ID {row[0]}: {row_err}")

            turso_conn.commit()
            turso_cursor.execute("SELECT COUNT(*) FROM study_logs;")
            total_turso_count = turso_cursor.fetchone()[0]
            turso_conn.close()

        elif use_libsql_client:
            http_url = turso_url.replace("libsql://", "https://")
            client = libsql_client.create_client_sync(url=http_url, auth_token=turso_token)

            client.execute("""
                CREATE TABLE IF NOT EXISTS study_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source_filename TEXT,
                    image_base64 TEXT,
                    tashkeel_text TEXT,
                    full_translation TEXT,
                    verbs_json TEXT,
                    nouns_json TEXT,
                    particles_json TEXT,
                    deep_sarf_json TEXT
                )
            """)

            for row in local_rows:
                try:
                    res = client.execute("""
                        INSERT OR IGNORE INTO study_logs 
                        (id, timestamp, source_filename, image_base64, tashkeel_text, full_translation, verbs_json, nouns_json, particles_json, deep_sarf_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, list(row))
                    if res.rows_affected > 0:
                        migrated_count += 1
                    else:
                        skipped_count += 1
                except Exception as row_err:
                    print(f"[!] Error inserting record ID {row[0]}: {row_err}")

            res_count = client.execute("SELECT COUNT(*) FROM study_logs;")
            total_turso_count = res_count.rows[0][0]
            client.close()

        local_conn.close()

        print("\n" + "=" * 60)
        print("  Migration Summary")
        print("=" * 60)
        print(f"  - Local records:     {len(local_rows)}")
        print(f"  - Migrated:          {migrated_count}")
        print(f"  - Skipped/Existing:  {skipped_count}")
        print(f"  - Total in Turso DB: {total_turso_count}")
        print("=" * 60)
        print("[+] Migration completed successfully!")

    except Exception as e:
        print(f"[-] Migration failed with error: {e}")
        local_conn.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
