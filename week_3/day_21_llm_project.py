# ==============================================
# 🧠 DAG 21: MINI-PROJEKT - LLM + MCP DEMO
# ==============================================
# Vi ska simulera hur en LLM samarbetar med MCP och verktyg.
# Syftet är att förstå logiken och flödet mellan dessa komponenter.

# -------------------------
# 📘 Steg 1: Simulera "verktyg"
# -------------------------
# Verktygen är externa funktioner som LLM kan anropa via MCP.
# Vi gör två enkla verktyg:
#   1) calculator_tool  - gör beräkningar
#   2) database_tool    - hämtar "data" ur en fiktiv databas

def calculator_tool(expression):
    """Simulerar ett kalkylatorverktyg som räknar ut uttryck."""
    print("\n[MCP] ➕ Kalkylatorverktyget anropas med uttryck:", expression)
    try:
        result = eval(expression)
        print("[MCP] 🔢 Resultat från kalkylator:", result)
        return result
    except Exception as e:
        print("[MCP] ❌ Fel i kalkylatorverktyget:", e)
        return None


def database_tool(query):
    """Simulerar ett databasverktyg."""
    print("\n[MCP] 🗄️  Databasverktyget anropas med fråga:", query)
    fake_db = {
        "hr": {"mean_salary": 42000, "employees": 150},
        "it": {"mean_salary": 48000, "employees": 200}
    }
    if "hr" in query.lower():
        return fake_db["hr"]
    elif "it" in query.lower():
        return fake_db["it"]
    else:
        print("[MCP] ❌ Ingen matchning i databasen.")
        return None


# -------------------------
# 📘 Steg 2: Simulera LLM
# -------------------------
# Här låtsas vi att LLM "förstår" text och avgör vilket verktyg den behöver.
# I verkligheten gör en riktig LLM detta med hjälp av prompt och kontext.

def llm_process(prompt):
    print("\n🤖 [LLM] Tar emot användarfråga:", prompt)

    # Enkel regelbaserad logik (simulerar LLM:s bedömning)
    if any(word in prompt.lower() for word in ["beräkna", "summa", "räkna", "plus", "minus"]):
        print("[LLM] 🔍 Känner igen att användaren vill räkna något.")
        expression = prompt.split("beräkna")[-1].strip()
        return ("calculator", expression)
    
    elif "lön" in prompt.lower() or "anställda" in prompt.lower():
        print("[LLM] 📊 Känner igen att användaren frågar om databasinfo.")
        return ("database", prompt)
    
    else:
        print("[LLM] 💬 Ingen verktygsanvändning behövs – svarar direkt.")
        return ("text", "Jag kan tyvärr inte hjälpa med det just nu.")


# -------------------------
# 📘 Steg 3: MCP-styrning
# -------------------------
# MCP är bryggan som kopplar ihop LLM och rätt verktyg.
# Den lyssnar på LLM:s behov och skickar anropet till rätt "specialist".

def mcp_controller(prompt):
    tool, content = llm_process(prompt)

    if tool == "calculator":
        result = calculator_tool(content)
        return f"Resultatet är {result}."
    
    elif tool == "database":
        data = database_tool(content)
        if data:
            return f"I {content.upper()} är medellönen {data['mean_salary']} SEK med {data['employees']} anställda."
        else:
            return "Jag hittade ingen information i databasen."
    
    else:
        return content

# -------------------------
# 📘 Steg 4: Testa hela kedjan
# -------------------------
print("\n==================== DEMO ====================")
queries = [
    "Beräkna 45 + 32 / 2",
    "Hur många anställda finns i HR?",
    "Hej, hur mår du?"
]

for q in queries:
    print("\n---------------------------------------------")
    answer = mcp_controller(q)
    print("[SVAR TILL ANVÄNDARE]:", answer)
