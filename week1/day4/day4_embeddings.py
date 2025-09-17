import numpy as np

# ---------------------------------------------------------
# DAG 4 – Bygga första embedding-exempel (superpedagogisk kod)
# Vi gör en minimal demo av:
# 1) Ord → vektor (embedding)
# 2) Beräkna kosinuslikhet
# 3) Använda embeddings för enkel semantisk sökning
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1. Skapa en liten ordlista ("vocabulary")
# ---------------------------------------------------------
words = ["hund", "katt", "äpple", "banan"]
print("Steg 1: Ordlistan är:", words)

# ---------------------------------------------------------
# 2. Skapa slumpmässiga embeddings
#
# Embedding = en vektor som representerar ett ord i ett flerdimensionellt rum.
# Dimensionen (d) bestämmer hur många siffror vektorn har.
# I riktiga modeller (OpenAI) är d = 1536.
# Här använder vi d = 5 för att kunna skriva ut och förstå.
#
# Formel:
#   embedding(word) = vektor i R^d
#
# Exempel: embedding("hund") = [0.12, -0.44, 0.31, ...]
# ---------------------------------------------------------
embedding_dim = 5
embeddings = {word: np.random.rand(embedding_dim) for word in words}

print("\nSteg 2: Genererade embeddings (slumpade vektorer):")
for w, vec in embeddings.items():
    print(f"{w}: {vec}")

# ---------------------------------------------------------
# 3. Definiera kosinuslikhet
#
# Formel:
#   cos_sim(A, B) = (A · B) / (||A|| * ||B||)
#
# Där:
#   A · B = Σ (a_i * b_i)   (skalarprodukt) A⋅B=A1​⋅B1​+A2​⋅B2​+A3​⋅B3​+⋯+An​⋅Bn​
#   ||A|| = sqrt( Σ (a_i^2) ) (normen = längden av vektorn)
#
# Tolkning:
# - cos_sim = 1  → vektorerna pekar åt samma håll (mycket lika)
# - cos_sim = 0  → vektorerna är ortogonala (helt olika)
# - cos_sim = -1 → motsatt riktning (motsatser)
# ---------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    # Skalarprodukt
    dot_product = np.dot(vec_a, vec_b)

    # Normer (längder)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Slutlig likhet
    similarity = dot_product / (norm_a * norm_b)

    # Printouts för att se alla steg
    print("\n--- Beräkning av kosinuslikhet ---")
    print("Vektor A:", vec_a)
    print("Vektor B:", vec_b)
    print("Skalarprodukt (A·B) =", dot_product)
    print("Norm ||A|| =", norm_a)
    print("Norm ||B|| =", norm_b)
    print("Cosine similarity =", similarity)

    return similarity

# ---------------------------------------------------------
# 4. Testa parvisa likheter
# ---------------------------------------------------------
sim_hund_katt = cosine_similarity(embeddings["hund"], embeddings["katt"])
sim_hund_äpple = cosine_similarity(embeddings["hund"], embeddings["äpple"])

print("\nSteg 4: Resultat av parvisa jämförelser:")
print("Likhet hund ↔ katt =", sim_hund_katt)
print("Likhet hund ↔ äpple =", sim_hund_äpple)

# ---------------------------------------------------------
# 5. Enkel "semantisk sökning"
#
# Vi låtsas att en användare söker på ordet "katt".
# Idé: Vi hittar embedding för "katt", jämför med alla ord,
#      och sorterar resultaten efter högsta kosinuslikhet.
# ---------------------------------------------------------
query = "katt"
query_vec = embeddings[query]

print("\nSteg 5: Semantisk sökning efter ordet:", query)

scores = {}
for w, vec in embeddings.items():
    # Beräkna kosinuslikhet
    score = cosine_similarity(query_vec, vec)
    scores[w] = score
    
    # Skriv ut vilket ord vi testar och vilket score vi får
    print(f"Testar ord: {w}, kosinuslikhet med '{query}': {score}")

# Sortera resultaten
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\nResultat av semantisk sökning (mest lik först):")
for w, score in sorted_scores:
    print(f"{w}: {score}")