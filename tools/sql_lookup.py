import time
import re
import os
import sqlite3
from tools.base import ToolResult, FailureMode

# Local SQLite DB path (populated on first use with sample data)
DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/megaai_local.db")

SAMPLE_DATA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL,
    stock INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    quantity INTEGER,
    customer TEXT,
    order_date TEXT
);
INSERT OR IGNORE INTO products VALUES (1, 'Widget A', 'Electronics', 29.99, 150);
INSERT OR IGNORE INTO products VALUES (2, 'Widget B', 'Electronics', 49.99, 80);
INSERT OR IGNORE INTO products VALUES (3, 'Gadget X', 'Tools', 19.99, 200);
INSERT OR IGNORE INTO products VALUES (4, 'Gadget Y', 'Tools', 39.99, 60);
INSERT OR IGNORE INTO products VALUES (5, 'Book Z', 'Books', 9.99, 500);
INSERT OR IGNORE INTO orders VALUES (1, 1, 3, 'Alice', '2024-01-10');
INSERT OR IGNORE INTO orders VALUES (2, 2, 1, 'Bob', '2024-01-11');
INSERT OR IGNORE INTO orders VALUES (3, 3, 5, 'Alice', '2024-01-12');
INSERT OR IGNORE INTO orders VALUES (4, 1, 2, 'Carol', '2024-01-13');
INSERT OR IGNORE INTO orders VALUES (5, 5, 10, 'Bob', '2024-01-14');
"""

NL_TO_SQL_MAP = [
    (r"(list|show|get) all products", "SELECT * FROM products"),
    (r"(list|show|get) all orders", "SELECT * FROM orders"),
    (r"products in (.+) category", "SELECT * FROM products WHERE category = '{}'"),
    (r"orders by (.+)", "SELECT * FROM orders WHERE customer = '{}'"),
    (r"total (revenue|sales)", "SELECT SUM(p.price * o.quantity) as total_revenue FROM orders o JOIN products p ON o.product_id = p.id"),
    (r"cheapest product", "SELECT * FROM products ORDER BY price ASC LIMIT 1"),
    (r"most expensive product", "SELECT * FROM products ORDER BY price DESC LIMIT 1"),
    (r"products? with (low|less) stock", "SELECT * FROM products WHERE stock < 100"),
    (r"count (of )?products", "SELECT COUNT(*) as product_count FROM products"),
    (r"count (of )?orders", "SELECT COUNT(*) as order_count FROM orders"),
]


def _ensure_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SAMPLE_DATA_SQL)
    conn.commit()
    return conn


def _nl_to_sql(natural_language: str) -> str | None:
    nl = natural_language.lower().strip()
    for pattern, sql_template in NL_TO_SQL_MAP:
        m = re.search(pattern, nl)
        if m:
            if '{}' in sql_template and m.groups():
                return sql_template.format(m.group(1).strip().title())
            return sql_template
    return None


def sql_lookup(natural_language: str) -> ToolResult:
    start = time.time()

    if not natural_language or not natural_language.strip():
        return ToolResult(
            success=False,
            failure_mode=FailureMode.malformed_input,
            error_message="natural_language query must be a non-empty string.",
            latency_ms=0.0,
        )

    sql = _nl_to_sql(natural_language)
    if not sql:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.empty_results,
            error_message=f"Could not convert to SQL: '{natural_language}'",
            latency_ms=(time.time() - start) * 1000,
        )

    try:
        conn = _ensure_db()
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        latency = (time.time() - start) * 1000

        if not rows:
            return ToolResult(
                success=False,
                failure_mode=FailureMode.empty_results,
                error_message="Query returned no results.",
                data={"sql": sql, "rows": [], "columns": columns},
                latency_ms=latency,
            )

        return ToolResult(
            success=True,
            data={
                "sql": sql,
                "columns": columns,
                "rows": [dict(zip(columns, row)) for row in rows],
                "row_count": len(rows),
            },
            latency_ms=latency,
        )
    except Exception as e:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.execution_error,
            error_message=str(e),
            latency_ms=(time.time() - start) * 1000,
        )
