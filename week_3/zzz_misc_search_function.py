"""
Experiment: Hur fungerar re.search()?

Syfte:
Visa hur re.search() hittar mönster i text med olika typer av regler.
Vi testar vanliga regex-symboler och förklarar resultatet tydligt.
"""

import re

# =============================
# STEG 1 – Exempeltexter
# =============================
texts = [
    "Antal HR anställda i företaget?",
    "Hur många jobbar på HR?",
    "Visa antal supportpersonal.",
    "Vad är medellönen för IT-avdelningen?",
    "Vilka började år 2024?"
]

# =============================
# STEG 2 – Exempel 1: Enkel sökning
# =============================
print("\n=== EXEMPEL 1: Enkel sökning efter ordet 'HR' ===\n")

for t in texts:
    if re.search(r"HR", t):
        print(f"✅ '{t}'  → matchade (innehåller 'HR')")
    else:
        print(f"❌ '{t}'  → ingen match")

# =============================
# STEG 3 – Exempel 2: Kombinera ord med '.*'
# =============================
print("\n=== EXEMPEL 2: Leta efter ordet 'antal' följt av 'HR' ===")
print("Förklaring: '.*' betyder 'valfritt antal tecken emellan'.\n")

pattern = r"antal.*HR"

for t in texts:
    if re.search(pattern, t, re.IGNORECASE):
        print(f"✅ '{t}'  → matchade mönstret '{pattern}'")
    else:
        print(f"❌ '{t}'  → ingen match för '{pattern}'")

# =============================
# STEG 4 – Exempel 3: Leta efter siffror
# =============================
print("\n=== EXEMPEL 3: Leta efter årtal (siffror) ===")
print("Förklaring: '\\d+' betyder 'en eller flera siffror'.\n")

for t in texts:
    match = re.search(r"\d+", t)
    if match:
        print(f"✅ '{t}'  → hittade siffror: {match.group()}")
    else:
        print(f"❌ '{t}'  → inga siffror hittade")

# =============================
# STEG 5 – Exempel 4: Leta efter meningar som börjar med visst ord
# =============================
print("\n=== EXEMPEL 4: Börjar texten med 'Visa'? ===")
print("Förklaring: '^Visa' betyder att texten måste börja med ordet 'Visa'.\n")

for t in texts:
    if re.search(r"^Visa", t):
        print(f"✅ '{t}'  → börjar med 'Visa'")
    else:
        print(f"❌ '{t}'  → börjar inte med 'Visa'")

# =============================
# STEG 6 – Exempel 5: Leta efter ord i slutet
# =============================
print("\n=== EXEMPEL 5: Slutar texten med frågetecken (?) ===")
print("Förklaring: '\\?$' betyder att sista tecknet i texten ska vara '?'.\n")

for t in texts:
    if re.search(r"\?$", t):
        print(f"✅ '{t}'  → slutar med frågetecken")
    else:
        print(f"❌ '{t}'  → slutar inte med frågetecken")

# =============================
# STEG 7 – Extra: Kombinera flera villkor
# =============================
print("\n=== EXEMPEL 6: Innehåller både 'HR' och siffror ===")
print("Förklaring: Vi söker två olika mönster med AND-logik.\n")

for t in texts:
    if re.search(r"HR", t, re.IGNORECASE) and re.search(r"\d+", t):
        print(f"✅ '{t}'  → innehåller både 'HR' och siffror")
    else:
        print(f"❌ '{t}'  → innehåller inte båda delarna")

print("\n🔚 Experiment klart! Lek gärna med mönstren ovan och ändra texterna för att se hur regex reagerar.")
