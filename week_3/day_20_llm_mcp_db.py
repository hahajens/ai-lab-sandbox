"""
DAG 20 – DEMO: LLM + Databasfråga via MCP

Syfte:
Visa hur en språkmodell (LLM) kan användas för att hämta data från en databas
genom att automatiskt översätta naturliga språkfrågor till SQL-kommandon.

Vi simulerar MCP-flödet:
  Användare (text) → LLM (tolkar) → MCP (skickar SQL) → Databas → Resultat
"""

import sqlite3
import re

# ========================
# STEG 1 – Skapa databasen
# ========================

# Vi använder SQLite – en filbaserad databas som kräver ingen server
conn = sqlite3.connect(':memory:')  # ":memory:" betyder att databasen ligger i RAM
cursor = conn.cursor()

# Skapa en enkel tabell för anställda
cursor.execute('''
CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    department TEXT,
    salary INTEGER,
    start_year INTEGER
)
''')

# Stoppa in testdata
employees = [
    ("Anna", "HR", 42000, 2023),
    ("Björn", "Support", 38000, 2022),
    ("Cecilia", "HR", 45000, 2024),
    ("David", "IT", 50000, 2021),
    ("Emma", "Support", 39000, 2023),
    ("Filip", "IT", 55000, 2020)
]
cursor.executemany('INSERT INTO employees (name, department, salary, start_year) VALUES (?, ?, ?, ?)', employees)
conn.commit()

print("✅ Databasen skapad och fylld med exempeldata!\n")

# ============================================
# STEG 2 – Simulerad LLM som förstår naturligt språk
# ============================================

def llm_to_sql(user_question):
    """
    Enkel simulering av hur en LLM kan omvandla språk till SQL.
    I verkligheten skulle MCP skicka frågan till ett SQL-verktyg.
    """

    print(f"🧠 LLM tolkar frågan: '{user_question}'")

    # Mycket enkel regelbaserad översättning (i verkligheten används LLM med MCP)
    if re.search(r"antal.*hr", user_question.lower()):
        sql = "SELECT COUNT(*) FROM employees WHERE department='HR';"
    elif re.search(r"antal.*support", user_question.lower()):
        sql = "SELECT COUNT(*) FROM employees WHERE department='Support';"
    elif re.search(r"lön.*it", user_question.lower()):
        sql = "SELECT AVG(salary) FROM employees WHERE department='IT';"
    elif re.search(r"började.*2024", user_question.lower()):
        sql = "SELECT name FROM employees WHERE start_year=2024;"
    else:
        sql = None

    if sql:
        print(f"🤖 Genererad SQL-fråga: {sql}\n")
    else:
        print("⚠️ Frågan kunde inte tolkas till SQL.\n")

    return sql

# ============================================
# STEG 3 – MCP simulerar anropet till databasen
# ============================================

def run_query(sql):
    """Kör SQL-frågan och returnerar resultat."""
    cursor.execute(sql)
    return cursor.fetchall()

# ============================================
# STEG 4 – Testa hela flödet
# ============================================

user_questions = [
    "Hur många personer jobbar på HR?",
    "Vad är medellönen för IT-avdelningen?",
    "Vilka började år 2024?",
    "Hur många jobbar i supporten?"
]

for q in user_questions:
    sql = llm_to_sql(q)
    if sql:
        result = run_query(sql)
        print(f"📊 Resultat från databasen: {result}\n")
    print("-" * 60)

# ============================================
# STEG 5 – Stäng databasen
# ============================================
conn.close()
print("\n🔚 Databasen stängd. Demo klar.")
