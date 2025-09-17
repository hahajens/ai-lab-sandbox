"""
Dag 03 – Embeddings: text -> siffror (simulerad)
Syfte:
  - Förstå vad embeddings är och hur vi mäter likhet mellan ord.
  - Lära sig beräkna kosinuslikhet och tolka resultaten.
Lärandemål:
  - Kunna skapa egna embeddings (simulerade) och jämföra dem.
  - Visualisera relationer mellan ord i 2D.
Förkunskaper:
  - Dag 1-2: Vektorer, skalarprodukt, kosinuslikhet.
Körinstruktioner:
  - Python 3.10+
  - pip install numpy matplotlib scikit-learn
  - Kör: python Dag03_embeddings_simulerad.py
Notera:
  - All förklaring finns i kommentarer och via print().
  - Lösningar till övningar finns längst ner och visas endast om show_solutions=True.
"""

# ============================================================
# DEL 0: Setup
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Seed för reproducerbarhet
# anväds till vektorer
np.random.seed(42)

# ============================================================
# DEL 1: Snabb repetition av vektorer
# ============================================================
print("\n===== DEL 1: Geometriska vektorer =====\n")

# Vi definierar några enkla 2D-vektorer
v1 = np.array([3, 4])   # punkt i planet
v2 = np.array([1, 0])   # enhetsvektor längs x-axeln
v3 = np.array([0, 1])   # enhetsvektor längs y-axeln

print("Vektor v1 =", v1)
print("Vektor v2 =", v2)
print("Vektor v3 =", v3)

# ============================================================
# Exempel: vektorer, längd, skalarprodukt och kosinuslikhet
# ============================================================

# Vi har två vektorer v1 och v2/v3 definierade tidigare
# v1 = [3,4]  → ett exempel på en vektor i 2D
# v2 = [1,0], v3 = [0,1]  → enhetsvektorer längs x- och y-axeln

# --- 1) Beräkna längd (norm) av v1 ---
# Normen (längden) av en vektor v definieras som:
# ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)
# Den visar "storleken" eller "magnituden" på vektorn.
length_v1 = np.linalg.norm(v1)  # np.linalg.norm räknar ut sqrt(sum(v_i^2))
print("\nLängden av v1 = sqrt(3^2 + 4^2) =", length_v1)
# Varför vi använder det: Normen används för att normalisera vektorer
# eller för att beräkna kosinuslikhet, som mäter riktning utan att påverkas av längd.

# --- 2) Skalarprodukt (dot product) mellan v2 och v3 ---
# Skalarprodukten definieras som:
# v·w = v[0]*w[0] + v[1]*w[1] + ... + v[n]*w[n]
# Den ger ett mått på hur "lika riktade" två vektorer är.
dot_v2_v3 = np.dot(v2, v3)  # np.dot beräknar summan av produkter av motsvarande komponenter
print("Skalarprodukt v2·v3 =", dot_v2_v3)
# Här är v2 och v3 vinkelräta → dot product = 0
print("Eftersom v2 och v3 står vinkelrätt blir skalarprodukten 0")

# --- 3) Kosinuslikhet ---
# Kosinuslikhet mäter hur lika riktningen av två vektorer är, oberoende av längd:
# cos(theta) = (v·w) / (||v|| * ||w||)
# Resultat nära 1 → mycket lika riktningar, nära 0 → ortogonala, nära -1 → motsatta
cosine_v2_v3 = dot_v2_v3 / (np.linalg.norm(v2) * np.linalg.norm(v3))
print("Kosinuslikhet mellan v2 och v3 =", cosine_v2_v3)
# Eftersom v2 och v3 är ortogonala (90 grader) blir kosinuslikheten 0
print("0 betyder helt ortogonala vektorer.\n")

# --- Varför vi använder detta ---
# 1) Längden av vektorn används för normalisering och förståelse av storlek.
#       Längd = hur stor vektorn är.   
#       Normalisering = gör längden 1 för att jämföra riktning utan att längden stör.
#       Vi använder detta för att kunna räkna kosinuslikhet och jämföra vektorer på ett rättvist sätt.

# 2) Skalarprodukten ger en enkel riktning-/likhetsmätning mellan vektorer.
#       Skalarprodukten mäter hur mycket två vektorer pekar i samma riktning:
#       Varje ord representeras som en vektor.
#       Dot product ger en första hint om hur lika orden är i betydelse.
#       Vi kombinerar ofta med normalisering → kosinuslikhet, så att längden inte påverkar.

# 3) Kosinuslikheten är standardmåttet för "likhet" i embeddingvärlden.
#       När vi jämför ord eller dokument med embeddings vill vi veta
#       hur lika deras riktning i vektorrummet är, oberoende av storlek.


# ============================================================
# DEL 2: Simulerade embeddings
# ============================================================
print("\n===== DEL 2: Simulerade embeddings =====\n")

# Lista med ord
words = ["hund", "katt", "bil", "lärare", "pedagog"]

# Skapa simuleringar av embeddings med små skillnader
# Vi simulerar att ord som är nära i betydelse får liknande vektorer
embeddings = {}

np.random.normal(size=1536)

# Skapar 1536 slumpmässiga tal från en normalfördelning (bell curve).
# Medelvärde = 0, standardavvikelse = 1 (default)
# Varje tal är “lite slumpmässigt”, men spridda kring 0.
# scale = standardavvikelsen för normalfördelningen.
# scale=0.01 → talen varierar mycket lite kring 0, alltså små “justeringar” eller “brus”.
# Vi använder det för att skapa ord som är nära varandra i betydelse, t.ex. katt nära hund:

# "hund" som bas
embeddings["hund"] = np.random.normal(size=1536)
# "katt" nära "hund"
embeddings["katt"] = embeddings["hund"] + np.random.normal(scale=0.01, size=1536)
# "bil" helt annan kategori
embeddings["bil"] = np.random.normal(size=1536)
# "lärare" bas
embeddings["lärare"] = np.random.normal(size=1536)
# "pedagog" nära "lärare"
embeddings["pedagog"] = embeddings["lärare"] + np.random.normal(scale=0.01, size=1536)

# Visa dimension och första 10 tal
for w in words:
    print(f"\nOrd: {w}")
    print("Embeddingens dimension:", len(embeddings[w]))
    print("Första 10 talen:", embeddings[w][:10])

# ============================================================
# DEL 3: Likhetsmått (kosinus)
# ============================================================
print("\n===== DEL 3: Kosinuslikhet mellan ord =====\n")

def cosine_similarity(a, b):
    """
    Beräkna kosinuslikhet mellan två vektorer a och b.
    Kosinuslikhet = (a·b) / (||a||*||b||)
    Steg-för-steg med printout av mellanvärden.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    print(f"Dot-product = {dot:.3f}, norm_a = {norm_a:.3f}, norm_b = {norm_b:.3f}")
    return dot / (norm_a * norm_b)

# Hund vs katt
sim_hund_katt = cosine_similarity(embeddings["hund"], embeddings["katt"])
print("Kosinuslikhet (hund, katt) =", sim_hund_katt)

# Hund vs bil
sim_hund_bil = cosine_similarity(embeddings["hund"], embeddings["bil"])
print("Kosinuslikhet (hund, bil) =", sim_hund_bil)

# Lärare vs pedagog
sim_teacher = cosine_similarity(embeddings["lärare"], embeddings["pedagog"])
print("Kosinuslikhet (lärare, pedagog) =", sim_teacher)

print("\nTolkning:")
print("- Hund & katt nära i betydelse → hög likhet")
print("- Hund & bil → låg likhet")
print("- Lärare & pedagog → hög likhet\n")

# ============================================================
# DEL 4: Visualisering (PCA)
# ============================================================
print("\n===== DEL 4: Visualisering av embeddings (2D) =====\n")

# Embeddings som matris
matrix = np.array([embeddings[w] for w in words])

# PCA till 2D
# Vi använder PCA (Principal Component Analysis) för att reducera dimensioner.
# PCA hittar de två riktningarna (dimensionerna) som fångar mest variation i datan.
#   Alltså:
#       Högdimensionell data → hitta “huvudaxlar” → projicera ner i 2D
#       Liknande vektorer hamnar nära varandra i 2D
#       Olika vektorer hamnar längre bort

pca = PCA(n_components=2)
reduced = pca.fit_transform(matrix)

# Print 2D-koordinater
for i, w in enumerate(words):
    print(w, "->", reduced[i])

# Rita ut orden i 2D
plt.figure(figsize=(6,6))
for i, w in enumerate(words):
    x, y = reduced[i]
    plt.scatter(x, y, marker="o", color="blue")
    plt.text(x+0.02, y+0.02, w, fontsize=12)

plt.title("Visualisering av embeddings (2D)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# ============================================================
# DEL 5: Övningar för eleven
# ============================================================
print("\n===== ÖVNINGAR =====\n")
print("1) Beräkna kosinuslikheten mellan 'bil' och 'pedagog'.")
print("2) Skapa en ny simulering för ordet 'kattunge' nära 'katt'. Beräkna kosinuslikheten med 'katt' och 'hund'.\n")

# ============================================================
# DEL 6: Lösningar (visas endast om flagga är True)
# ============================================================
show_solutions = True

if show_solutions:
    print("LÖSNING ÖVNING 1:")
    sim_bil_ped = cosine_similarity(embeddings["bil"], embeddings["pedagog"])
    print("Kosinuslikhet (bil, pedagog) =", sim_bil_ped)
    
    print("\nLÖSNING ÖVNING 2:")
    emb_kattunge = embeddings["katt"] + np.random.normal(scale=0.01, size=1536)
    sim_katt_kattunge = cosine_similarity(embeddings["katt"], emb_kattunge)
    sim_hund_kattunge = cosine_similarity(embeddings["hund"], emb_kattunge)
    print("Kosinuslikhet (katt, kattunge) =", sim_katt_kattunge)
    print("Kosinuslikhet (hund, kattunge) =", sim_hund_kattunge)

# ============================================================
# DEL 7: Slutsats
# ============================================================
print("\n===== SLUTSATS =====\n")
print("Embeddings är vektorer med semantisk betydelse.")
print("Kosinuslikhet visar hur lika två ord är i betydelse.")
print("Simulering visar principen – med riktiga modeller blir värden mer exakta.\n")
