# ===============================================================
# DAG 19: LLM + MCP + Kalkylator
# ===============================================================
# Syfte: Visa hur en Large Language Model (LLM) kan använda
# Model Context Protocol (MCP) för att anropa ett externt verktyg
# (t.ex. en kalkylator) på ett säkert och strukturerat sätt.
# ===============================================================

# 🧩 1. Kalkylator-verktyget
# ---------------------------------------------------------------
# Detta är vårt externa "verktyg" som MCP ska anropa.
# Det tar emot en operation (t.ex. "add") och två tal.
# ---------------------------------------------------------------

def calculator_tool(operation: str, a: float, b: float) -> float:
    """
    Enkel kalkylator-funktion som utför en matematisk operation.
    operation: str - "add", "subtract", "multiply", "divide"
    a, b: float - talen att beräkna
    return: float - resultatet
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else float('inf')
    else:
        raise ValueError("Okänd operation")

# 🧠 Exempel:
print("\n[TEST] 2 + 3 =", calculator_tool("add", 2, 3))
print("[TEST] 10 * 5 =", calculator_tool("multiply", 10, 5))


# 🧩 2. MCP-controller
# ---------------------------------------------------------------
# MCP fungerar som en mellanhand mellan LLM och verktyg.
# Den tar emot ett "verktygsanrop" och hanterar det säkert.
# ---------------------------------------------------------------

def mcp_controller(request: dict) -> dict:
    """
    Simulerar MCP:s roll: tar emot ett request från LLM,
    validerar, anropar rätt verktyg och returnerar resultat.
    """
    print("\n[MCP] Tar emot förfrågan från LLM:", request)

    # Kontrollera att request har rätt format
    if "tool" not in request or "params" not in request:
        return {"error": "Ogiltig MCP-förfrågan"}

    tool_name = request["tool"]
    params = request["params"]

    # Här definierar vi vilka verktyg som MCP känner till:
    if tool_name == "calculator":
        try:
            result = calculator_tool(**params)
            print("[MCP] Verktyget 'calculator' kördes korrekt.")
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": f"Okänt verktyg: {tool_name}"}


# 🧩 3. LLM-simulator
# ---------------------------------------------------------------
# Här låtsas vi att LLM själv inser att den behöver använda
# kalkylatorn via MCP. I praktiken sker detta automatiskt.
# ---------------------------------------------------------------

def llm_simulator(user_input: str) -> str:
    """
    Simulerar hur en LLM tänker: tolkar frågan, avgör om
    ett verktyg behövs, skapar MCP-request och tolkar svaret.
    """
    print("\n[LLM] Användarfråga:", user_input)

    # Enkel texttolkning (i verkligheten används NLP och parsing)
    if "sum" in user_input or "+" in user_input or "add" in user_input:
        # Extrahera siffror (enkel parsing)
        import re
        numbers = [float(x) for x in re.findall(r'\d+', user_input)]
        if len(numbers) == 2:
            a, b = numbers
            # Skapa MCP-request
            request = {
                "tool": "calculator",
                "params": {"operation": "add", "a": a, "b": b}
            }
            response = mcp_controller(request)
            if "result" in response:
                return f"Summan av {a} och {b} är {response['result']}."
            else:
                return f"Ett fel uppstod: {response['error']}"
    return "Jag kan tyvärr inte beräkna det utan MCP-verktyg."


# 🧩 4. Testkörning
# ---------------------------------------------------------------
print("\n========== DEMO ==========")
print(llm_simulator("What is the sum of 347 and 89?"))
print(llm_simulator("Add 15 + 9"))
print(llm_simulator("Multiply 7 and 8"))  # Ej stödd parsing
