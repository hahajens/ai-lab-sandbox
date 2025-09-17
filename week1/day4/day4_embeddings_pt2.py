import numpy as np

# -----------------------------
# 1. Skapa "databas" av ord och embeddings
# -----------------------------
# Vi använder små dimensioner (5) för enkel demonstration
embeddings = {
    "katt": np.array([0.9, 0.1, 0.3, 0.7, 0.2]),
    "hund": np.array([0.8, 0.2, 0.4, 0.6, 0.3]),
    "äpple": np.array([0.1, 0.9, 0.8, 0.2, 0.5]),
    "bil": np.array([0.3, 0.7, 0.1, 0.9, 0.4])
}

print("Embeddings-databas:")
for word, vec in embeddings.items():
    print(f"{word}: {vec}")

# -----------------------------
# 2. Funktion för kosinuslikhet
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
# -----------------------------
def cosine_similarity(A, B):
    # Skalarprodukt
    dot_product = np.dot(A, B)
    
    # Vektorlängder
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Kosinuslikhet
    cos_sim = dot_product / (norm_A * norm_B)
    
    # Print av alla delberäkningar
    print("\nBeräkning av kosinuslikhet:")
    print("Vektor A:", A)
    print("Vektor B:", B)
    print("Skalarprodukt (A·B) =", dot_product)
    print("Längd |A| =", norm_A)
    print("Längd |B| =", norm_B)
    print("Kosinuslikhet =", cos_sim)
    
    return cos_sim

# -----------------------------
# 3. Semantisk sökning
# -----------------------------
query = "katt"
query_vec = embeddings[query]

print("\n--- Steg 3: Semantisk sökning efter ordet:", query, "---")

scores = {}
for w, vec in embeddings.items():
    # Beräkna kosinuslikhet
    score = cosine_similarity(query_vec, vec)
    scores[w] = score
    
    # Print för varje ord
    print(f"\nTestar ord: {w}, kosinuslikhet med '{query}': {score}")

# -----------------------------
# 4. Sortera och visa resultat
# -----------------------------
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n--- Resultat av semantisk sökning (mest lik först) ---")
for w, score in sorted_scores:
    print(f"{w}: {score}")
