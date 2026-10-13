import os
import sys
import json
import time
import re
import asyncio
import sqlite3
import pandas as pd
try:
    import pytest
except ImportError:
    class _MockPytestMark:
        @staticmethod
        def asyncio(fn):
            return fn
    class _MockPytest:
        mark = _MockPytestMark()
    pytest = _MockPytest()

sys.path.insert(0, os.path.abspath("."))

from app.database import (
    init_db, reset_db, get_batches, get_leads,
    check_and_book_slot, bulk_upload_leads, DB_PATH
)
from app.knowledge_base import (
    knowledge_engine, get_system_context, update_custom_rules
)
from app.call_agent import (
    JanbrightCallAgent, extract_action, simulate_outbound_dial
)

def test_knowledge_base_ingestion():
    """Verify knowledge ingestion loads .md, .json, and .xlsx enterprise context."""
    summary = knowledge_engine.get_summary()
    assert len(summary["md_files"]) > 0, "No markdown files loaded in knowledge base"
    assert len(summary["json_files"]) > 0, "No JSON files loaded in knowledge base"
    assert len(summary["xlsx_files"]) > 0, "No Excel files loaded in knowledge base"

    context = get_system_context()
    assert "Janbright World School" in context
    assert "Mokila" in context
    assert "Cambridge International" in context or "CAIE" in context
    assert "Robotics" in context
    assert "Archery" in context
    assert "Swimming" in context
    assert "STRICT FEE DEFLECTION" in context
    assert "2 sentences" in context

    # Test dynamic custom rule updates
    test_rule = "All visitors must register at Mokila Gate 1 before entering the campus."
    update_custom_rules(test_rule)
    updated_context = get_system_context()
    assert test_rule in updated_context

def test_database_initialization_and_mock_data():
    """Verify relational ledger schema and required mock data seeds."""
    reset_db()
    batches = get_batches()
    assert len(batches) >= 6, "Expected at least 6 batches seeded"

    batch_dict = {(b["activity"], b["slot_time"]): b for b in batches}
    
    # Check required seed numbers from Directive
    assert batch_dict[("Robotics", "Saturday 10:30 AM")]["max_seats"] == 15
    assert batch_dict[("Robotics", "Saturday 10:30 AM")]["booked_seats"] == 14  # 1 left

    assert batch_dict[("Robotics", "Saturday 03:00 PM")]["max_seats"] == 15
    assert batch_dict[("Robotics", "Saturday 03:00 PM")]["booked_seats"] == 5

    assert batch_dict[("Archery", "Saturday 10:30 AM")]["max_seats"] == 10
    assert batch_dict[("Archery", "Saturday 10:30 AM")]["booked_seats"] == 10  # FULL

    assert batch_dict[("Archery", "Saturday 03:00 PM")]["max_seats"] == 10
    assert batch_dict[("Archery", "Saturday 03:00 PM")]["booked_seats"] == 3

    assert batch_dict[("Swimming", "Saturday 10:30 AM")]["max_seats"] == 12
    assert batch_dict[("Swimming", "Saturday 10:30 AM")]["booked_seats"] == 4

    # Check 3 sample parents from Mokila
    leads = get_leads()
    assert len(leads) >= 3, "Expected at least 3 initial sample leads"
    archery_conflict = [l for l in leads if l["activity"] == "Archery" and l["slot_time"] == "Saturday 10:30 AM"]
    assert len(archery_conflict) > 0, "Expected Mokila parent who requested full Archery 10:30 AM batch"
    assert archery_conflict[0]["status"] == "WAITLIST_FULL"

def test_capacity_checks_and_deterministic_booking():
    """Verify capacity rejection with alternate offer and booking decrement."""
    reset_db()

    # 1. Try booking Archery at Saturday 10:30 AM (Full 10/10)
    res_full = check_and_book_slot(
        parent_name="Vikram Varma",
        phone="9988776655",
        child="Rohan Varma",
        grade="Grade 8",
        activity="Archery",
        slot="Saturday 10:30 AM"
    )
    assert res_full["success"] is False
    assert res_full["status"] == "FULL"
    assert res_full["alternative_slot"] == "Saturday 03:00 PM"
    assert "at full capacity" in res_full["message"]
    assert "Saturday 03:00 PM" in res_full["message"]

    # 2. Try booking Robotics at Saturday 10:30 AM (14/15 -> becomes 15/15)
    res_avail = check_and_book_slot(
        parent_name="Srinivas Rao",
        phone="9849012345",
        child="Aarav Rao",
        grade="Grade 6",
        activity="Robotics",
        slot="Saturday 10:30 AM"
    )
    assert res_avail["success"] is True
    assert res_avail["status"] == "CONFIRMED"
    assert res_avail["booked_seats"] == 15
    assert res_avail["remaining_seats"] == 0

    # 3. Next booking for Robotics at Saturday 10:30 AM must now be REJECTED (now 15/15 full!)
    res_now_full = check_and_book_slot(
        parent_name="Anjali Nair",
        phone="9849099999",
        child="Varun Nair",
        grade="Grade 6",
        activity="Robotics",
        slot="Saturday 10:30 AM"
    )
    assert res_now_full["success"] is False
    assert res_now_full["status"] == "FULL"
    assert res_now_full["alternative_slot"] == "Saturday 03:00 PM"

def test_bulk_lead_upload():
    """Verify ingestion of prospective leads from pandas DataFrame."""
    test_data = pd.DataFrame([
        {
            "Parent Name": "Manoj Reddy",
            "Contact Phone": "9812345678",
            "Student Name": "Pranav Reddy",
            "Grade": "Grade 7",
            "Workshop": "Robotics",
            "Slot Time": "Saturday 03:00 PM"
        },
        {
            "Parent Name": "Deepa Joshi",
            "Contact Phone": "9823456789",
            "Student Name": "Isha Joshi",
            "Grade": "Grade 5",
            "Workshop": "Swimming",
            "Slot Time": "Saturday 10:30 AM"
        }
    ])
    count = bulk_upload_leads(test_data)
    assert count == 2

    leads = get_leads(limit=10)
    parents = [l["parent_name"] for l in leads]
    assert "Manoj Reddy" in parents
    assert "Deepa Joshi" in parents

def test_action_parser():
    """Verify JSON extraction from ACTION: markup."""
    sample_text = 'ACTION: {"tool": "book_trial", "args": {"parent_name": "Srinivas", "phone": "9849012345", "child_name": "Aarav", "grade": "Grade 6", "activity": "Robotics", "slot": "Saturday 10:30 AM"}} Confirmed for Aarav!'
    json_str, full_str = extract_action(sample_text)
    assert json_str is not None
    data = json.loads(json_str)
    assert data["tool"] == "book_trial"
    assert data["args"]["child_name"] == "Aarav"
    assert data["args"]["activity"] == "Robotics"

@pytest.mark.asyncio
async def test_agent_fee_deflection_and_call_simulation():
    """Verify fee query deflection and automated call turn simulation."""
    reset_db()
    res = await simulate_outbound_dial(
        phone_number="9849012345",
        scenario="fee_deflection_and_robotics",
        parent_name="Srinivas Rao",
        child_name="Aarav",
        grade="Grade 6"
    )
    assert res["status"] == "COMPLETED"
    assert len(res["transcript"]) >= 4

    # Verify fee deflection in agent response
    fee_query_turn = [t for t in res["transcript"] if "tuition" in t["text"].lower() or "fee" in t["text"].lower()]
    assert len(fee_query_turn) > 0, "Expected parent turn asking about tuition fees"

    # Verify no academic tuition number (e.g. rupees, lakhs, currency) was given
    for turn in res["transcript"]:
        if turn["speaker"] == "Janbright Voice Agent":
            reply_lower = turn["text"].lower()
            assert not re.search(r'\blakh\b', reply_lower)
            assert "per annum" not in reply_lower
            assert not re.search(r'\brs\.?\s*\d+', reply_lower)
            assert not re.search(r'\b\d+\s*(?:lakh|thousand|rupee|inr)', reply_lower)
            assert "₹" not in reply_lower

    # Verify ledger was mutated
    action_res = res["action_result"]
    assert action_res is not None
    assert action_res["status"] == "CONFIRMED"
    assert action_res["booked_seats"] == 15

if __name__ == "__main__":
    test_knowledge_base_ingestion()
    test_database_initialization_and_mock_data()
    test_capacity_checks_and_deterministic_booking()
    test_bulk_lead_upload()
    test_action_parser()
    asyncio.run(test_agent_fee_deflection_and_call_simulation())
    print("\n>>> ALL JANBRIGHT PLATFORM UNIT & INTEGRATION TESTS PASSED 100%! <<<\n")
