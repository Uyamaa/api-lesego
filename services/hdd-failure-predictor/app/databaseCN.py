import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_latest_reading(drive_id):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM get_latest_reading(%s)", (drive_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row

def check_connection():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"), connect_timeout=3)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False
