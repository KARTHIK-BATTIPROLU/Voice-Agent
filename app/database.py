import sqlite3
import os
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

DB_PATH = os.path.abspath("school_enterprise.db")
_db_lock = threading.Lock()

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH, force_reseed: bool = False):
    """Initializes tables and seeds initial data if empty or forced."""
    with _db_lock:
        conn = get_connection(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            activity TEXT NOT NULL,
            slot_time TEXT NOT NULL,
            max_seats INTEGER NOT NULL,
            booked_seats INTEGER NOT NULL,
            PRIMARY KEY (activity, slot_time)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS parent_leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_name TEXT,
            phone_number TEXT,
            child_name TEXT,
            grade TEXT,
            activity TEXT,
            slot_time TEXT,
            status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

        cur.execute("SELECT COUNT(*) as count FROM batches")
        batch_count = cur.fetchone()["count"]

        if batch_count == 0 or force_reseed:
            if force_reseed:
                cur.execute("DELETE FROM batches")
                cur.execute("DELETE FROM parent_leads")
                conn.commit()

            # Seed Batches strictly as specified in Directive
            seed_batches = [
                ("Robotics", "Saturday 10:30 AM", 15, 14),   # 1 seat left
                ("Robotics", "Saturday 03:00 PM", 15, 5),    # 10 seats left
                ("Archery",  "Saturday 10:30 AM", 10, 10),   # FULL - triggers alternate offer
                ("Archery",  "Saturday 03:00 PM", 10, 3),    # 7 seats left
                ("Swimming", "Saturday 10:30 AM", 12, 4),    # 8 seats left
                ("Swimming", "Saturday 03:00 PM", 12, 2)     # 10 seats left
            ]
            cur.executemany(
                "INSERT OR REPLACE INTO batches (activity, slot_time, max_seats, booked_seats) VALUES (?, ?, ?, ?)",
                seed_batches
            )

            # Seed 3 sample parents from Mokila (including Archery 10:30 AM capacity conflict)
            seed_leads = [
                ("Srinivas Rao", "9849012345", "Aarav Rao", "Grade 6", "Robotics", "Saturday 10:30 AM", "CONFIRMED", "2026-10-05 09:15:00"),
                ("Sunita Reddy", "9849098765", "Ananya Reddy", "Grade 4", "Swimming", "Saturday 10:30 AM", "CONFIRMED", "2026-10-05 10:20:00"),
                ("Vikram Varma", "9988776655", "Rohan Varma", "Grade 8", "Archery", "Saturday 10:30 AM", "WAITLIST_FULL", "2026-10-05 11:45:00")
            ]
            cur.executemany(
                "INSERT INTO parent_leads (parent_name, phone_number, child_name, grade, activity, slot_time, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                seed_leads
            )
            conn.commit()
            print(f"[Database] Initialized and seeded ledger at {db_path}.", flush=True)

        conn.close()

def normalize_activity(activity_str: str) -> str:
    """Normalizes activity string to match batch records."""
    if not activity_str:
        return "Robotics"
    act_lower = str(activity_str).strip().lower()
    if "robot" in act_lower:
        return "Robotics"
    elif "arch" in act_lower:
        return "Archery"
    elif "swim" in act_lower:
        return "Swimming"
    return activity_str.strip().title()

def normalize_slot(slot_str: str) -> str:
    """Normalizes slot time string to match standard batch schedule."""
    if not slot_str:
        return "Saturday 10:30 AM"
    s_lower = str(slot_str).strip().lower()
    if "10:30" in s_lower or "10" in s_lower or "morning" in s_lower:
        return "Saturday 10:30 AM"
    if "3:00" in s_lower or "03:00" in s_lower or "3pm" in s_lower or "3 pm" in s_lower or "afternoon" in s_lower or "15:00" in s_lower or "3" in s_lower:
        return "Saturday 03:00 PM"
    return "Saturday 10:30 AM"

def check_and_book_slot(
    parent_name: str = "Prospective Parent",
    phone: Optional[str] = None,
    child: Optional[str] = None,
    grade: str = "Grade 5",
    activity: str = "Robotics",
    slot: Optional[str] = None,
    db_path: str = DB_PATH,
    **kwargs
) -> Dict[str, Any]:
    """
    Enforces deterministic concurrency using SQLite transaction locking.
    If seats are full, rejects with alternative suggestion.
    If available, increments booked_seats, inserts lead record with status 'CONFIRMED', and returns status.
    Supports keyword argument aliases (child_name, slot_time, phone_number, caller_name).
    """
    clean_child_arg = child or kwargs.get("child_name") or "Student"
    clean_phone_arg = phone or kwargs.get("phone_number") or "Unspecified"
    clean_slot_arg = slot or kwargs.get("slot_time") or kwargs.get("time_slot") or "Saturday 10:30 AM"
    clean_parent_arg = parent_name or kwargs.get("caller_name") or "Prospective Parent"
    clean_grade_arg = grade or kwargs.get("class") or "Grade 5"

    norm_activity = normalize_activity(activity)
    norm_slot = normalize_slot(clean_slot_arg)
    clean_parent = str(clean_parent_arg).strip() if clean_parent_arg else "Prospective Parent"
    clean_phone = str(clean_phone_arg).strip() if clean_phone_arg else "Unspecified"
    clean_child = str(clean_child_arg).strip() if clean_child_arg else "Student"
    clean_grade = str(clean_grade_arg).strip() if clean_grade_arg else "Grade 5"

    with _db_lock:
        conn = get_connection(db_path)
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "SELECT activity, slot_time, max_seats, booked_seats FROM batches WHERE activity = ? AND slot_time = ?",
                (norm_activity, norm_slot)
            )
            row = cur.fetchone()

            if not row:
                conn.rollback()
                return {
                    "success": False,
                    "status": "REJECTED",
                    "activity": norm_activity,
                    "slot": norm_slot,
                    "message": f"Workshop for {norm_activity} at {norm_slot} was not found."
                }

            max_seats = row["max_seats"]
            booked_seats = row["booked_seats"]

            # CAPACITY CHECK
            if booked_seats >= max_seats:
                # Find alternate available slot for this activity
                cur.execute(
                    "SELECT slot_time, max_seats, booked_seats FROM batches WHERE activity = ? AND slot_time != ? AND booked_seats < max_seats",
                    (norm_activity, norm_slot)
                )
                alt_row = cur.fetchone()
                alt_slot = alt_row["slot_time"] if alt_row else "Saturday 03:00 PM"

                # Record lead as waitlisted / capacity rejection
                cur.execute(
                    """
                    INSERT INTO parent_leads (parent_name, phone_number, child_name, grade, activity, slot_time, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'WAITLIST_FULL')
                    """,
                    (clean_parent, clean_phone, clean_child, clean_grade, norm_activity, norm_slot)
                )
                conn.commit()

                return {
                    "success": False,
                    "status": "FULL",
                    "activity": norm_activity,
                    "slot": norm_slot,
                    "booked_seats": booked_seats,
                    "max_seats": max_seats,
                    "alternative_slot": alt_slot,
                    "message": (
                        f"The {norm_activity} workshop for {norm_slot} is currently at full capacity ({booked_seats}/{max_seats}). "
                        f"We have open spots on {alt_slot}, would you like me to reserve that instead?"
                    )
                }

            # SEAT AVAILABLE - CONFIRM BOOKING
            new_booked = booked_seats + 1
            cur.execute(
                "UPDATE batches SET booked_seats = ? WHERE activity = ? AND slot_time = ?",
                (new_booked, norm_activity, norm_slot)
            )

            cur.execute(
                """
                INSERT INTO parent_leads (parent_name, phone_number, child_name, grade, activity, slot_time, status)
                VALUES (?, ?, ?, ?, ?, ?, 'CONFIRMED')
                """,
                (clean_parent, clean_phone, clean_child, clean_grade, norm_activity, norm_slot)
            )
            lead_id = cur.lastrowid
            conn.commit()

            remaining = max_seats - new_booked
            return {
                "success": True,
                "status": "CONFIRMED",
                "lead_id": lead_id,
                "activity": norm_activity,
                "slot": norm_slot,
                "booked_seats": new_booked,
                "max_seats": max_seats,
                "remaining_seats": remaining,
                "message": (
                    f"Confirmed! Complimentary trial for {clean_child} in {norm_activity} on {norm_slot} is reserved at Janbright World School, Mokila."
                )
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "status": "ERROR",
                "message": f"Ledger error: {str(e)}"
            }
        finally:
            conn.close()

def _clean_cell(val: Any, default: str = "") -> str:
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    if s.lower() in ["nan", "none", "null", ""]:
        return default
    return s

def bulk_upload_leads(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Ingests prospective parent contact lists from pandas DataFrame."""
    if df is None or df.empty:
        return 0

    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        if any(k in c_clean for k in ["parent", "father", "mother", "guardian"]):
            col_map["parent_name"] = col
        elif any(k in c_clean for k in ["phone", "mobile", "contact", "cell"]):
            col_map["phone_number"] = col
        elif any(k in c_clean for k in ["child", "student", "kid"]):
            col_map["child_name"] = col
        elif any(k in c_clean for k in ["grade", "class", "standard"]):
            col_map["grade"] = col
        elif any(k in c_clean for k in ["activity", "workshop", "course"]):
            col_map["activity"] = col
        elif any(k in c_clean for k in ["slot", "time", "batch"]):
            col_map["slot_time"] = col

    with _db_lock:
        conn = get_connection(db_path)
        cur = conn.cursor()
        inserted = 0

        for _, row in df.iterrows():
            parent = _clean_cell(row.get(col_map.get("parent_name", "parent_name")), default="Parent")
            phone = _clean_cell(row.get(col_map.get("phone_number", "phone_number")), default="")
            child = _clean_cell(row.get(col_map.get("child_name", "child_name")), default="Student")
            grade = _clean_cell(row.get(col_map.get("grade", "grade")), default="Grade 5")
            act = normalize_activity(_clean_cell(row.get(col_map.get("activity", "activity")), default="Robotics"))
            slot = normalize_slot(_clean_cell(row.get(col_map.get("slot_time", "slot_time")), default="Saturday 10:30 AM"))
            status = _clean_cell(row.get("status"), default="NEW")

            cur.execute(
                """
                INSERT INTO parent_leads (parent_name, phone_number, child_name, grade, activity, slot_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (parent, phone, child, grade, act, slot, status)
            )
            inserted += 1

        conn.commit()
        conn.close()
        return inserted

def get_batches(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all workshop batches and capacity status."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT activity, slot_time, max_seats, booked_seats FROM batches ORDER BY activity, slot_time")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_leads(limit: int = 100, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves parent leads ordered by newest first."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, parent_name, phone_number, child_name, grade, activity, slot_time, status, timestamp FROM parent_leads ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def reset_db(db_path: str = DB_PATH):
    """Resets database to factory seed data."""
    init_db(db_path=db_path, force_reseed=True)

# Auto-initialize on import
init_db()
