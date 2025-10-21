# ============================================================
# 📘 DAG 17: Prompting-grunder med praktiska exempel
# ============================================================
# Den här lektionen visar hur olika typer av PROMPTS påverkar
# hur en LLM (Large Language Model) svarar.
#
# Vi använder här en simulerad "modell" (en enkel Python-funktion)
# så du kan se strukturen och effekten utan API-anrop.
# ============================================================

# Vi börjar med att definiera en enkel "modell" som svarar
# baserat på nyckelord i prompten.
# I verkligheten hade vi använt t.ex. OpenAI GPT-5 API.
def fake_llm(prompt: str) -> str:
    """
    En fejkad LLM-modell som returnerar olika svar beroende på prompt-innehåll.
    Vi gör detta för att illustrera hur prompten styr svaret.
    """
    prompt_lower = prompt.lower()

    if "zero-shot" in prompt_lower:
        return "Zero-shot: Modellen försöker gissa direkt utan exempel."
    elif "one-shot" in prompt_lower:
        return "One-shot: Modellen lär sig mönstret från ett exempel och följer det."
    elif "few-shot" in prompt_lower:
        return "Few-shot: Modellen generaliserar från flera exempel till en ny fråga."
    elif "chain-of-thought" in prompt_lower:
        return "Chain-of-Thought: Jag tänker steg för steg: 2 + 2 = 4."
    elif "professor" in prompt_lower:
        return "Som professor i AI: En embedding är ett sätt att representera ord som siffror i ett flerdimensionellt rum."
    else:
        return "Jag försöker svara utifrån min förståelse av frågan: 'AI är när datorer kan utföra uppgifter som normalt kräver mänsklig intelligens.'"


# ============================================================
# 🧩 DEL 1: Zero-shot prompting
# ============================================================
# Zero-shot = ingen exempeldata, bara instruktion.
# Modellen måste förstå uppgiften direkt.
# ============================================================
prompt_zero = "Zero-shot: Förklara kort vad en vektor är."
print("🧩 Zero-shot prompt:")
print(prompt_zero)
print("🧠 Modellens svar:")
print(fake_llm(prompt_zero))
print("-" * 60)


# ============================================================
# 🧩 DEL 2: One-shot prompting
# ============================================================
# One-shot = du visar ETT exempel så modellen lär sig formatet.
# Exempel: vi visar hur en förklaring ska se ut.
# ============================================================
prompt_one = """
One-shot:
Exempel: En hund är ett djur som ofta är husdjur.
Nu: En katt är...
"""
print("🧩 One-shot prompt:")
print(prompt_one)
print("🧠 Modellens svar:")
print(fake_llm(prompt_one))
print("-" * 60)


# ============================================================
# 🧩 DEL 3: Few-shot prompting
# ============================================================
# Few-shot = flera exempel så modellen kan generalisera.
# Här visar vi flera exempel i samma format.
# ============================================================
prompt_few = """
Few-shot:
Exempel:
Hund -> Animal
Katt -> Animal
Blomma -> Plant
Träd -> Plant
Nu: Fisk ->
"""
print("🧩 Few-shot prompt:")
print(prompt_few)
print("🧠 Modellens svar:")
print(fake_llm(prompt_few))
print("-" * 60)


# ============================================================
# 🧩 DEL 4: Chain-of-Thought (CoT) prompting
# ============================================================
# CoT = be modellen "tänka högt" steg för steg.
# Det hjälper den att resonera logiskt.
# ============================================================
prompt_cot = """
Chain-of-Thought:
Fråga: Om jag har 2 äpplen och köper 3 till, hur många har jag då?
Tänk steg för steg.
"""
print("🧩 Chain-of-Thought prompt:")
print(prompt_cot)
print("🧠 Modellens svar:")
print(fake_llm(prompt_cot))
print("-" * 60)


# ============================================================
# 🧩 DEL 5: Role prompting
# ============================================================
# Role prompting = ge modellen en roll eller persona.
# Det påverkar ton, stil och komplexitet i svaret.
# ============================================================
prompt_role = """
Du är en professor i AI.
Förklara vad en embedding är på ett sätt som en nybörjare förstår.
"""
print("🧩 Role prompt:")
print(prompt_role)
print("🧠 Modellens svar:")
print(fake_llm(prompt_role))
print("-" * 60)


# ============================================================
# 🧩 DEL 6: Jämförelse – tydlig vs vag prompt
# ============================================================
# Här visar vi hur mycket skillnad det gör att vara tydlig.
# ============================================================
prompt_vag = "Förklara AI."
prompt_tydlig = "Du är en AI-lärare. Förklara vad AI är för en 12-åring, använd enkla ord och ett exempel."

print("🧩 Jämförelse mellan vag och tydlig prompt:")
print("VAG PROMPT:", prompt_vag)
print("🧠 Svar:", fake_llm(prompt_vag))
print("\nTYDLIG PROMPT:", prompt_tydlig)
print("🧠 Svar:", fake_llm(prompt_tydlig))
print("-" * 60)


# ============================================================
# 🧩 DEL 7: Summering av prompting-principer
# ============================================================
# Vi sammanfattar de viktigaste lärdomarna från dagen.
# ============================================================
print("📘 SAMMANFATTNING DAG 17:")
print("""
1️⃣ En prompt är instruktionen som styr modellens beteende.
2️⃣ Zero-shot = ingen exempeldata.
3️⃣ One-/Few-shot = ge exempel så modellen lär sig formatet.
4️⃣ Chain-of-Thought = be modellen resonera stegvis.
5️⃣ Role prompting = sätt en roll (lärare, expert, utvecklare).
6️⃣ Tydliga, kontextuella och exempelrika prompts ger bäst resultat.
7️⃣ Alltid: specificera roll, format, ton och målgrupp.
""")
