# -----------------------------
# DAG 7: Mini-projekt – Semantisk sökning med riktiga embeddings + PCA-visualisering
# -----------------------------
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# 1) Skapa "databas" av meningar
# -----------------------------
texts = [
    "Jag älskar att köra bil.",
    "Katter är mysiga husdjur.",
    "AI är framtiden för teknik.",
    "Jag spelade fotboll igår.",
    "Hundar är trogna vänner."
]

print("Databas-meningar:")
for t in texts:
    print("-", t)

# -----------------------------
# 2) Skapa embeddings med en färdig modell
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
all_embeddings = model.encode(texts)
print("\nEmbeddings shape:", all_embeddings.shape)
print("Exempel embedding (första mening):", all_embeddings[0][:5], "...")  # första 5 dimensioner

# -----------------------------
# 3) Centrera embeddings
# -----------------------------
# Målet med centrering: 
# När vi skapar embeddings för meningar hamnar varje vektor på olika ställen i det höga
# dimensionella rummet. Vissa dimensioner kan vara "generellt stora" över alla meningar.
# Om vi inte centrera riskerar dessa generella biasar att påverka likhetsberäkningen.
#
# Centrering betyder att vi flyttar alla vektorer så att medelvärdet av varje dimension blir 0.
# Formeln för centrering:
#   x_centered[i] = x[i] - mean_vector
# där mean_vector är medelvärdet över alla meningar för varje dimension:
#   mean_vector[j] = (1/n) * sum_{i=1}^{n} x[i,j]
# n = antal meningar, j = dimension
# -----------------------------

# Beräkna medelvärde för varje dimension
mean_vector = np.mean(all_embeddings, axis=0)  
# Förklaring:
# all_embeddings.shape = (5, 384)  # 5 meningar, 384 dimensioner
# np.mean(..., axis=0) summerar alla 5 meningars värden för varje dimension
# och delar med 5 (antal meningar). Resultatet blir en vektor med 384 element.
# Den representerar "centrum" för alla meningar i vektorrummet.

# Subtrahera medelvärdet från varje embedding (centrera)
centered = all_embeddings - mean_vector  
# Förklaring:
# Varje mening flyttas i vektorrummet så att centrum ligger i origo (0,0,...,0)
# Detta tar bort bias som är gemensam för alla meningar.
# Formellt: om x är en mening i 384D, x_centered = x - mean_vector

# Visualisering av effekten (för de första 5 dimensionerna)
print("\nMedelvärde per dimension (första 5):", mean_vector[:5])
print("Exempel: första mening före/efter centrering (första 5 dimensioner):")
print("Före:", all_embeddings[0][:5])
print("Efter:", centered[0][:5])

# -----------------------------
# 4) Normalisera vektorer (längd = 1)
# -----------------------------
norms = np.linalg.norm(centered, axis=1, keepdims=True)
normalized = centered / norms
print("\nNormer efter centrering:", norms.flatten())
print("Exempel normaliserad embedding (första mening):", normalized[0][:5])

# -----------------------------
# 5) Skapa query och embedding
# -----------------------------
query = "Jag gillar hundar."
query_embedding = model.encode([query])[0]
query_centered = query_embedding - mean_vector
query_normalized = query_centered / np.linalg.norm(query_centered)

print("\nQuery:", query)
print("Query embedding (första 5):", query_embedding[:5])
print("Query efter centrering:", query_centered[:5])
print("Query normaliserad:", query_normalized[:5])

# -----------------------------
# 6) Beräkna kosinuslikhet
# -----------------------------
cosine_similarities = np.dot(normalized, query_normalized)
print("\nKosinuslikheter mellan query och alla meningar:", cosine_similarities)

# -----------------------------
# 7) Hitta mest liknande mening
# -----------------------------
best_idx = np.argmax(cosine_similarities)
print("\nMest liknande mening:")
print(texts[best_idx])
print("Kosinuslikhet:", cosine_similarities[best_idx])

# -----------------------------
# 8) PCA-visualisering (2D)
# -----------------------------
# Kombinera alla embeddings + query
all_vectors_2d = np.vstack([normalized, query_normalized])
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(all_vectors_2d)

plt.figure(figsize=(8,6))
for i, text in enumerate(texts):
    plt.scatter(vectors_2d[i,0], vectors_2d[i,1], color='blue')
    plt.text(vectors_2d[i,0]+0.01, vectors_2d[i,1]+0.01, text, fontsize=9)

# Markera query
plt.scatter(vectors_2d[-1,0], vectors_2d[-1,1], color='red', label='Query')
plt.text(vectors_2d[-1,0]+0.01, vectors_2d[-1,1]+0.01, query, fontsize=9, color='red')

plt.title("PCA 2D-visualisering av meningar + query")
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.legend()
plt.show()
