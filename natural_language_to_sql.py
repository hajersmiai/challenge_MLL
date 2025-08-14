import os
import sys
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import argparse
#from tabulate import tabulate

# Optional LLM providers
# OpenAI (native)
try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None

# Gemini via LangChain
try:
    from langchain.chat_models import init_chat_model  # pip install langchain google-generativeai langchain-google-genai
except Exception:
    init_chat_model = None
# Load .env    
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
DB_PATH = os.getenv("NL2SQL_DB_PATH", "kbo_database.db")
LLM_PROVIDER = os.getenv("NL2SQL_LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("NL2SQL_OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("NL2SQL_GEMINI_MODEL", "gemini-2.0-flash-lite")

MAX_ROWS = int(os.getenv("NL2SQL_MAX_ROWS", "500"))
TIMEOUT_SECS = int(os.getenv("NL2SQL_SQL_TIMEOUT", "30"))


# -----------------------------
# Utilities: Schema Introspection
# -----------------------------
def list_tables(conn: sqlite3.Connection) -> List[str]:
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
    """
    cur = conn.execute(q)
    return [r[0] for r in cur.fetchall()]


def table_columns(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str]]:
    # returns list of (name, type)
    q = f"PRAGMA table_info({table});"
    cur = conn.execute(q)
    return [(r[1], r[2]) for r in cur.fetchall()]


def build_schema_prompt(conn: sqlite3.Connection) -> str:
    tables = list_tables(conn)
    lines = []
    for t in tables:
        cols = table_columns(conn, t)
        cols_str = ", ".join([f"{c} {typ or ''}".strip() for c, typ in cols])
        lines.append(f"- {t}({cols_str})")
    schema_text = "\n".join(lines)
    return schema_text


# -----------------------------
# Prompt Template
# -----------------------------
BASE_SYSTEM_PROMPT = (
    "You are a careful SQL assistant. Given a SQLite database schema and a natural "
    "language request, produce a single valid SQL query that answers it.\n"
    "Rules:\n"
    "- Only use the given schema (tables/columns).\n"
    "- Use SQLite dialect.\n"
    "- Prefer explicit JOINs with ON.\n"
    "- If aggregation is used, include GROUP BY as needed.\n"
    "- Limit result size reasonably when appropriate.\n"
    "- Return only the SQL, no explanations.\n"
)

USER_PROMPT_TEMPLATE = (
    "SCHEMA:\n{schema}\n\n"
    "QUESTION:\n{question}\n\n"
    "Return ONLY the SQL query enclosed in a markdown SQL block, like:\n"
    "```sql\nSELECT ...\n```\n"
)


# -----------------------------
# LLM Clients
# -----------------------------
class SQLGenerator:
    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider.lower()
        self._ensure_client()

    def _ensure_client(self):
        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not available. Install and set OPENAI_API_KEY.")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY missing in environment/.env.")
            self.client = OpenAI()
        elif self.provider == "gemini":
            if init_chat_model is None:
                raise RuntimeError("LangChain + Google GenAI not available. Install and set GOOGLE_API_KEY.")
            if not os.getenv("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY missing in environment/.env.")
            # LangChain unified init
            self.model = init_chat_model(GEMINI_MODEL, model_provider="google_genai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_sql(self, schema: str, question: str) -> str:
        prompt = USER_PROMPT_TEMPLATE.format(schema=schema, question=question)

        if self.provider == "openai":
            msg = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msg,
                temperature=0,
                max_tokens=800,
            )
            content = resp.choices[0].message.content
        else:  # gemini via langchain
            content = self.model.invoke([
                ("system", BASE_SYSTEM_PROMPT),
                ("human", prompt),
            ]).content

        sql = extract_sql_from_text(content)
        return sql


# -----------------------------
# SQL Extraction & Validation
# -----------------------------
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def extract_sql_from_text(text: str) -> str:
    m = SQL_BLOCK_RE.search(text or "")
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = (text or "").strip()

    # Trim trailing semicolons and whitespace
    candidate = candidate.strip().rstrip(";").strip()
    return candidate


DANGEROUS_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
    "attach", "detach", "pragma", "vacuum",
]
FORBIDDEN_TABLES = ["sqlite_master"]

def validate_sql(sql: str):
    low = sql.lower()
    # Allow CTEs with WITH ... SELECT ... but only SELECT queries
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Only SELECT (or WITH ... SELECT) queries are allowed.")

    # Single statement only
    if ";" in sql:
        raise ValueError("Multiple statements are not allowed.")

    # Block dangerous keywords and sqlite internals
    if any(kw in low for kw in DANGEROUS_KEYWORDS):
        raise ValueError("Only read-only queries are allowed (found DDL/DML keywords).")

    if any(t in low for t in FORBIDDEN_TABLES):
        raise ValueError("Access to SQLite internal tables is not allowed.")


# -----------------------------
# Execution
# -----------------------------
@dataclass
class NL2SQLResult:
    question: str
    sql: str
    df: pd.DataFrame


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """
    Open SQLite safely in read-only. Works on Windows paths, too.
    Falls back to query-only mode if the URI approach fails.
    """
    p = Path(db_path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found at: {p}")

    # Preferred: true read-only via URI + absolute POSIX path
    uri = f"file:{p.as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True, timeout=TIMEOUT_SECS)
    except sqlite3.OperationalError:
        # Fallback: normal open + forbid writes
        conn = sqlite3.connect(str(p), timeout=TIMEOUT_SECS)
        # Prevent DDL/DML during this connection
        conn.execute("PRAGMA query_only = ON;")
        return conn

def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    # Apply LIMIT if missing to avoid runaway results (soft-guard)
    low = sql.lower()
    if "limit " not in low and "count(" not in low:
        sql = f"{sql}\nLIMIT {MAX_ROWS}"
    return pd.read_sql_query(sql, conn)


# -----------------------------
# Public API
# -----------------------------
def answer_question(question: str, db_path: Optional[str] = None, provider: Optional[str] = None) -> NL2SQLResult:
    db_path = db_path or DB_PATH
    provider = (provider or LLM_PROVIDER).lower()

    with connect_readonly(db_path) as conn:
        schema = build_schema_prompt(conn)

    generator = SQLGenerator(provider=provider)
    sql = generator.generate_sql(schema=schema, question=question)
    validate_sql(sql)

    with connect_readonly(db_path) as conn:
        df = run_sql(conn, sql)

    return NL2SQLResult(question=question, sql=sql, df=df)


from tabulate import tabulate

# Change this if your DB file is named differently
DB = DB_PATH  # e.g., "Chinook_Sqlite.sqlite"

TEST_QUESTIONS = [
    # kbo_database examples 
    "Geographic Distribution of Companies in Belgium by Zipcode",
    "Top 10 of Juridical Form Distribution with Enterprise Type.",
    "Creation of Enterprises by Type (Last 10 Years).",
    "Dominant Economic Sector per Postal Code / Province.",
    "Top 10 most mature sectors (NACE)",
    "Top 10 newest sectors (NACE)"
]

def show(q):
    res = answer_question(q, db_path=DB)
    print("\n" + "="*80)
    print("Q:", q)
    print("SQL:\n", res.sql)
    print("RESULTS:")
    print(tabulate(res.df, headers="keys", tablefmt="psql", showindex=False))



# --- Streamlit is optional at runtime (only required for UI mode) ---
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# -----------------------------
# Helpers
# -----------------------------
def print_result_to_console(question: str, db_path: str, provider: str):
    """Runs the NL→SQL pipeline and prints SQL + results in the terminal."""
    res = answer_question(question, db_path=db_path, provider=provider)

    print("\n" + "=" * 80)
    print("Q:", question)
    print("\nSQL:")
    print(res.sql)
    print("\nRESULTS:")
    if res.df.empty:
        print("[No rows]")
    else:
        print(tabulate(res.df, headers="keys", tablefmt="psql", showindex=False))


def in_streamlit_runtime() -> bool:
    """Best-effort detection if we're being run by Streamlit."""
    try:
        if st is None:
            return False
        # Streamlit ≥1.30 exposes st.runtime.exists()
        exists = getattr(st, "runtime", None)
        if exists and getattr(st.runtime, "exists", None):
            return st.runtime.exists()
    except Exception:
        pass
    # Fallback env heuristic
    return os.environ.get("STREAMLIT_SERVER_PORT") is not None


# -----------------------------
# CLI mode
# -----------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Natural Language → SQL on SQLite (CLI)")
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"SQLite path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--provider",
        choices=["auto","openai","gemini"],
        default=os.environ.get("NL2SQL_LLM_PROVIDER", "openai"),
        help="LLM provider (default: env NL2SQL_LLM_PROVIDER or 'openai')",
    )
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (ignored by Streamlit).")
    parser.add_argument("question", nargs="*", help="Natural language question")
    args = parser.parse_args()

    q = " ".join(args.question).strip() if args.question else ""
    if not q:
        try:
            q = input("Enter your question: ").strip()
        except EOFError:
            print("[ERROR] No question provided via arg or stdin.")
            sys.exit(1)

    try:
        print_result_to_console(q, db_path=args.db, provider=args.provider)
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)


# -----------------------------
# Streamlit UI mode
# -----------------------------
def main_streamlit():
    if st is None:
        print("[ERROR] Streamlit is not installed. Run `pip install streamlit` or use CLI mode.")
        sys.exit(1)

    st.title("Natural Language → SQL (SQLite)")
    question = st.text_input("Ask a question about the database:")
    db = st.text_input("SQLite path:", DB_PATH)
    default_provider = os.environ.get("NL2SQL_LLM_PROVIDER", "auto")
    provider = st.selectbox("LLM Provider", ["auto", "openai", "gemini"], index=["auto","openai","gemini"].index(default_provider if default_provider in ["auto","openai","gemini"] else "auto"))


    run = st.button("Run")
    if run and question.strip():
        try:
            res = answer_question(question, db_path=db, provider=provider)
            st.code(res.sql, language="sql")
            st.dataframe(res.df)
        except Exception as e:                   
            msg = str(e)
            if "insufficient_quota" in msg or "Error code: 429" in msg:
                st.error("OpenAI quota exceeded (429). Choose 'gemini' as provider or add credits. If you've enabled 'auto', set GOOGLE_API_KEY for fallback.")
            else:
                st.error(msg)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # If we're inside Streamlit, show the UI; otherwise default to CLI
    # (You can always force CLI with `--cli` when invoking with plain Python.)
    if in_streamlit_runtime():
        main_streamlit()
    else:
        main_cli()



