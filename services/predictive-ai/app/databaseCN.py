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