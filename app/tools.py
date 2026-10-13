import sqlite3
import json
from pydantic import BaseModel, Field

# Initialize local ledger
conn = sqlite3.connect("enterprise_ledger.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_name TEXT,
    phone_number TEXT,
    date TEXT,
    time_slot TEXT,
    status TEXT
)
""")
conn.commit()

class BookingArgs(BaseModel):
    caller_name: str = Field(description="Name of the patient or client")
    phone_number: str = Field(description="10-digit mobile number")
    date: str = Field(description="Date formatted as YYYY-MM-DD or relative like tomorrow")
    time_slot: str = Field(description="Time of booking, e.g., 10:00 AM or 4:00 PM")

def book_appointment(caller_name: str, phone_number: str, date: str, time_slot: str) -> str:
    """Executes a confirmed booking directly into the SQLite enterprise ledger."""
    cursor.execute(
        "INSERT INTO appointments (caller_name, phone_number, date, time_slot, status) VALUES (?, ?, ?, ?, ?)",
        (caller_name, phone_number, date, time_slot, "CONFIRMED")
    )
    conn.commit()
    print(f"\n[LEDGER TRANSACTION]: Confirmed {caller_name} for {date} at {time_slot}.")
    return f"Confirmed for {caller_name} on {date} at {time_slot}."
