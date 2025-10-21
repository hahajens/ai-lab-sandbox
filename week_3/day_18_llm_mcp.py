# ============================================
# DAG 18 – MCP (Model Context Protocol) Demo
# ============================================
# Vi simulerar hur en LLM anropar ett verktyg (calculator) via MCP
# Steg för steg, med tydliga printouts och förklaringar
# ============================================

import json

# 1️⃣ Låt oss först skapa en "LLM-förfrågan"
# I verkligheten skickar en LLM JSON-data via MCP till en verktygsserver.
# Vi simulerar detta som ett Python-dict.

tool_request = {
    "type": "tool_call",        # signalerar att LLM vill använda ett verktyg
    "tool": "calculator",       # vilket verktyg?
    "input": {"expression": "(34 + 6) * 2"}  # indata till verktyget
}

print("=== LLM skickar MCP-förfrågan ===")
print(json.dumps(tool_request, indent=2))
print()

# 2️⃣ Verktygssidan (MCP-servern) tar emot detta anrop.
# Den ser vilket verktyg som efterfrågas och skickar vidare till rätt funktion.

def calculator_tool(input_data):
    """
    Enkel kalkylator-funktion som evaluerar matematiska uttryck.
    OBS: I riktig MCP-kod används inte eval() direkt pga säkerhet!
    Här gör vi det bara för pedagogiskt exempel.
    """
    expression = input_data["expression"]
    print(f"🔧 Verktyget 'calculator' tar emot uttrycket: {expression}")
    
    # Utför beräkningen:
    result = eval(expression)
    
    # Returnera standardiserat MCP-svar:
    return {
        "type": "tool_result",
        "tool": "calculator",
        "output": {"result": result}
    }

# 3️⃣ MCP-servern "matchar" verktyget och exekverar det:
if tool_request["tool"] == "calculator":
    tool_response = calculator_tool(tool_request["input"])
else:
    tool_response = {"type": "error", "message": "Okänt verktyg"}

print("\n=== Verktyget svarar via MCP ===")
print(json.dumps(tool_response, indent=2))
print()

# 4️⃣ LLM tar emot svaret och använder det i sitt resonemang.
result_value = tool_response["output"]["result"]

print("=== LLM genererar svar till användaren ===")
print(f"Resultatet av uttrycket {tool_request['input']['expression']} är {result_value}.")
print()

# 5️⃣ Loggning – i riktiga system loggas alla MCP-anrop för säkerhet och spårbarhet.
log_entry = {
    "user": "demo_user",
    "tool": tool_request["tool"],
    "expression": tool_request["input"]["expression"],
    "result": result_value
}

print("=== Loggpost (för spårbarhet) ===")
print(json.dumps(log_entry, indent=2))
