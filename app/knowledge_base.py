import os
import glob
import json
from typing import Dict, Any, Optional
import pandas as pd

class KnowledgeEngine:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = os.path.abspath(knowledge_dir)
        self.md_docs: Dict[str, str] = {}
        self.json_docs: Dict[str, Any] = {}
        self.xlsx_docs: Dict[str, pd.DataFrame] = {}
        self.custom_rules: str = ""
        self.load_all()

    def load_all(self, directory: Optional[str] = None):
        """Loads all .md, .json, and .xlsx files from the knowledge folder."""
        if directory:
            self.knowledge_dir = os.path.abspath(directory)
        
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)

        self.md_docs.clear()
        self.json_docs.clear()
        self.xlsx_docs.clear()

        # Load Markdown files
        for md_path in glob.glob(os.path.join(self.knowledge_dir, "*.md")):
            filename = os.path.basename(md_path)
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.md_docs[filename] = content
                    if filename == "custom_rules.md":
                        self.custom_rules = content
            except Exception as e:
                print(f"[KnowledgeBase] Error loading {filename}: {e}", flush=True)

        # Load JSON files
        for json_path in glob.glob(os.path.join(self.knowledge_dir, "*.json")):
            filename = os.path.basename(json_path)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.json_docs[filename] = json.load(f)
            except Exception as e:
                print(f"[KnowledgeBase] Error loading {filename}: {e}", flush=True)

        # Load Excel files
        for xlsx_path in glob.glob(os.path.join(self.knowledge_dir, "*.xlsx")):
            filename = os.path.basename(xlsx_path)
            try:
                df = pd.read_excel(xlsx_path)
                self.xlsx_docs[filename] = df
            except Exception as e:
                print(f"[KnowledgeBase] Error loading {filename}: {e}", flush=True)

    def update_custom_rules(self, rules: str):
        """Updates runtime custom rules and persists to knowledge/custom_rules.md."""
        clean = rules.strip()
        # If user passed an entire prompt containing "ADDITIONAL CAMPUS DIRECTIVES:", extract directives
        if "ADDITIONAL CAMPUS DIRECTIVES:" in clean:
            parts = clean.split("ADDITIONAL CAMPUS DIRECTIVES:", 1)
            clean = parts[1].strip()
        
        self.custom_rules = clean
        custom_file = os.path.join(self.knowledge_dir, "custom_rules.md")
        with open(custom_file, "w", encoding="utf-8") as f:
            f.write(self.custom_rules)
        self.md_docs["custom_rules.md"] = self.custom_rules

    def get_system_context(self) -> str:
        """
        Formats enterprise context into a compact system instruction for Qwen-2.5-1.5B.
        Extracts loaded context dynamically from markdown and JSON documents.
        Constrained for low TTFT latency and strict policy compliance.
        """
        # Base school and curriculum facts from knowledge ingestion
        school_name = "Janbright World School"
        campus_loc = "Mokila, Hyderabad (near Financial District)"
        curriculum = "Cambridge International (CAIE) from Kindergarten to Grade 12"
        workshops_summary = "Robotics (Ages 7-14), Archery (Ages 9-16), Swimming (Ages 6-15)"
        slots_summary = "Every Saturday at 10:30 AM and 3:00 PM"
        cost_summary = "Complimentary trial workshop (100% Free)"

        # Inspect parsed weekend_workshops.json if available
        if "weekend_workshops.json" in self.json_docs:
            wk_data = self.json_docs["weekend_workshops.json"]
            if isinstance(wk_data, dict):
                school_name = wk_data.get("institution", school_name)
                campus_loc = wk_data.get("location", campus_loc)
                if "workshops" in wk_data and isinstance(wk_data["workshops"], list):
                    wk_items = []
                    for w in wk_data["workshops"]:
                        act = w.get("activity", "")
                        age = w.get("target_age", "")
                        if act and age:
                            wk_items.append(f"{act} ({age})")
                    if wk_items:
                        workshops_summary = ", ".join(wk_items)

        prompt = (
            f"You are the Voice Admissions Agent for {school_name}, Mokila, Hyderabad.\n"
            f"Campus: {campus_loc}.\n"
            f"Curriculum: {curriculum}.\n"
            f"Weekend Club Workshops: {workshops_summary}.\n"
            f"Trial Slots: {slots_summary}.\n"
            f"Cost: {cost_summary}.\n\n"
            "ABSOLUTE RULES:\n"
            "1. STRICT FEE DEFLECTION: Never state or quote academic tuition fee numbers over the phone. "
            "Politely defer fee details to an in-person on-campus admissions meeting and campus walkthrough at Mokila.\n"
            "2. LENGTH CONSTRAINT: Keep spoken responses strictly under 2 sentences. Be warm, natural, and concise. No markdown or bullet points.\n"
            "3. WORKSHOP FLOW:\n"
            "   - Turn 1: Warm greeting for Janbright World School, Mokila. Confirm parent name & introduce Saturday workshops.\n"
            "   - Turn 2: Activity selection (Robotics, Archery, Swimming) and child grade.\n"
            "   - Turn 3: Time slot selection (Saturday 10:30 AM vs 3:00 PM), deflecting any fee queries.\n"
            "   - Turn 4: Execute booking tool.\n"
            "4. TOOL EXECUTION:\n"
            "   When parent details (parent_name, phone, child_name, grade, activity, slot) are gathered, output:\n"
            '   ACTION: {"tool": "book_trial", "args": {"parent_name": "...", "phone": "...", "child_name": "...", "grade": "...", "activity": "...", "slot": "..."}}\n'
            "   Followed immediately by a short spoken confirmation under 2 sentences.\n"
        )
        if self.custom_rules:
            prompt += f"\nADDITIONAL CAMPUS DIRECTIVES:\n{self.custom_rules}\n"

        return prompt.strip()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "knowledge_dir": self.knowledge_dir,
            "md_files": list(self.md_docs.keys()),
            "json_files": list(self.json_docs.keys()),
            "xlsx_files": list(self.xlsx_docs.keys()),
            "custom_rules_active": bool(self.custom_rules),
            "custom_rules": self.custom_rules
        }

# Global instance for shared system state
knowledge_engine = KnowledgeEngine()

def get_system_context() -> str:
    """Exposed top-level context provider."""
    return knowledge_engine.get_system_context()

def load_knowledge_base(directory: str = "knowledge"):
    knowledge_engine.load_all(directory)

def update_custom_rules(rules: str):
    knowledge_engine.update_custom_rules(rules)

