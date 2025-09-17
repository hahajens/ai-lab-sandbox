# -----------------------------
# Dag 5: Enkel semantisk sökning med riktiga embeddings
# -----------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# 1) Ladda en färdig embedding-modell
# -----------------------------
print("Laddar modell 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modell laddad!\n")

# -----------------------------
# 2) Skapa databasen av meningar
# -----------------------------
texts = [
    "Jag älskar att köra bil.",
    "Katter är mysiga husdjur.",
    "AI är framtiden för teknik.",
    "Jag spelade fotboll igår.",
    "Hundar är trogna vänner."
]

query = "Jag gillar att köra bil."

print("Databas med meningar:")
for i, t in enumerate(texts):
    print(f"Mening {i}: {t}")
print("\nQuery:", query)
print("\n")

# -----------------------------
# 3) Skapa embeddings
# -----------------------------
print("Skapar embeddings för databasen...")
text_embeddings = model.encode(texts)
print("Embeddings för databasen skapade!\n")

print("Skapar embedding för query...")
query_embedding = model.encode([query])[0]
print("Embedding för query skapad!\n")

# -----------------------------
# 4) Visa exempel på vektorer
# -----------------------------
print("Exempel på embedding-vektorer:")
for i, emb in enumerate(text_embeddings):
    print(f"Mening {i} embedding (första 10 värden): {emb[:10]} ...")  # visa bara första 10 dimensioner
print("\nQuery embedding (första 10 värden):", query_embedding[:10], "...\n")

# -----------------------------
# 5) Kosinuslikhet-funktion
# -----------------------------
def cosine_similarity(a, b):
    """
    Beräknar kosinuslikhet mellan två vektorer a och b
    cos(a,b) = (a · b) / (||a|| * ||b||)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    
    # Print mellanresultat för pedagogik
    print(f"Dot-product: {dot_product:.4f}, ||a||: {norm_a:.4f}, ||b||: {norm_b:.4f} -> similarity: {similarity:.4f}")
    
    return similarity

# -----------------------------
# 6) Beräkna likhet mellan query och alla meningar
# -----------------------------
print("\nBeräknar kosinuslikhet mellan query och varje mening:")
similarities = []

for i, emb in enumerate(text_embeddings):
    print(f"\nJämför mening {i}: '{texts[i]}' med query '{query}'")
    sim = cosine_similarity(query_embedding, emb)
    similarities.append(sim)
    print(f"→ Kosinuslikhet: {sim:.4f}")

# -----------------------------
# 7) Sortera och visa mest liknande mening
# -----------------------------
most_similar_idx = np.argmax(similarities)

print("\n---- Resultat ----")
print(f"Mest liknande mening till query '{query}':")
print(f"'{texts[most_similar_idx]}'")
print(f"Kosinuslikhet: {similarities[most_similar_idx]:.4f}")

# -----------------------------
# 8) Visa alla meningar sorterade efter likhet
# -----------------------------
print("\nAlla meningar sorterade efter likhet (högst först):")
sorted_indices = np.argsort(similarities)[::-1]  # högst först
for idx in sorted_indices:
    print(f"'{texts[idx]}' → Kosinuslikhet: {similarities[idx]:.4f}")
