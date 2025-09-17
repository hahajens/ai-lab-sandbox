import numpy as np

# -----------------------------
# 1) Skapa en liten "databas" av meningar
# -----------------------------
texts = [
    "Jag älskar att köra bil.",
    "Katter är mysiga husdjur.",
    "AI är framtiden för teknik.",
    "Jag spelade fotboll igår.",
    "Hundar är trogna vänner."
]

print("Databasen med meningar:")
for t in texts:
    print("-", t)
print("\n")

# -----------------------------
# 2) Skapa en query
# -----------------------------
query = "Jag gillar att köra bil."
print("Query:", query)
print("\n")


# -----------------------------
# 3) Skapa embeddings (vektorer)
# -----------------------------
# Embeddings = sätt att representera meningar som vektorer (listor med siffror)
# Tanken: meningar som betyder ungefär samma sak får "liknande" vektorer

# -----------------------------
# 3a) Sätta seed för slumpgeneratorn
# -----------------------------
np.random.seed(42)  # låser slumpen så att vi alltid får samma resultat
print("Seed satt till 42 → samma slumpvärden varje gång vi kör koden\n")

# -----------------------------
# 3b) Bestäm dimensionen på embeddings
# -----------------------------
embedding_dim = 5  # antal siffror i varje embedding-vektor
print(f"Varje embedding-vektor kommer att ha {embedding_dim} dimensioner\n")

# -----------------------------
# 3c) Skapa embeddings för databasen (meningar)
# -----------------------------
# len(texts) = antal meningar i databasen
# np.random.rand(rows, columns) skapar en matris med slumpvärden mellan 0 och 1
text_embeddings = np.random.rand(len(texts), embedding_dim)

# Visa hur text_embeddings ser ut
print("Embeddings för varje mening i databasen:")
for i, emb in enumerate(text_embeddings):
    print(f"Mening {i} ({texts[i]}):")
    print("Vektor:", emb)
    print("Antal dimensioner:", len(emb))
    print("Summa av vektorn (bara för att se storlek):", np.sum(emb))
    print("-"*40)
print("\n")

# -----------------------------
# 3d) Skapa embedding för query
# -----------------------------
query_embedding = np.random.rand(embedding_dim)
print("Embedding för query:")
print(query_embedding)
print("Antal dimensioner i query-vektor:", len(query_embedding))
print("Summa av query-vektor:", np.sum(query_embedding))
print("\n")

# -----------------------------
# 3e) Förklara vad vi gjort
# -----------------------------
print("Sammanfattning:")
print("Vi har skapat:")
print(f"- {len(text_embeddings)} embeddings för meningarna i databasen, varje med {embedding_dim} dimensioner")
print("- 1 embedding för query, med samma dimension")
print("Dessa embeddings är nu klara för att beräkna likhet (t.ex. kosinuslikhet) mellan query och meningar")
# -----------------------------
# 4) Beräkna kosinuslikhet
# -----------------------------
def cosine_similarity(a, b):
    """Beräknar kosinuslikheten mellan två vektorer"""
    dot_product = np.dot(a, b)           # skalarprodukt
    norm_a = np.linalg.norm(a)           # längd av a
    norm_b = np.linalg.norm(b)           # längd av b
    similarity = dot_product / (norm_a * norm_b)  # kosinuslikhet
    return similarity
# -----------------------------
# 4) Beräkna likhet för varje mening med tydligare utskrift
# -----------------------------
similarities = []

print("Jämför query med varje mening i databasen:\n")
for i, emb in enumerate(text_embeddings):
    sim = cosine_similarity(emb, query_embedding)
    similarities.append(sim)
    print(f"Mening {i}: '{texts[i]}'")
    print(f"Query: '{query}'")
    print(f"Kosinuslikhet: {sim:.4f}\n")

# -----------------------------
# 5) Sortera och visa mest liknande mening med tydligare utskrift
# -----------------------------
most_similar_idx = np.argmax(similarities)
print("---- Resultat ----")
print(f"Mest liknande mening till query '{query}':")
print(f"'{texts[most_similar_idx]}'")
print(f"Kosinuslikhet: {similarities[most_similar_idx]:.4f}")
