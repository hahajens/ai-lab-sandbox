import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# -----------------------------
# DAG 6: PCA + t-SNE med pedagogiska printouts och fixade komplexa varningar
# -----------------------------

# 1. Skapa en liten "databas" av meningar
texts = [
    "Jag älskar att köra bil.",
    "Katter är mysiga husdjur.",
    "AI är framtiden för teknik.",
    "Jag spelade fotboll igår.",
    "Hundar är trogna vänner."
]

query = "Jag gillar sport."

# 2. Skapa embeddings med en färdig modell
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
query_embedding = model.encode([query])[0]

# Slå ihop embeddings + query
all_embeddings = np.vstack([embeddings, query_embedding])
print("Shape för all_embeddings (meningar + query):", all_embeddings.shape)

# -----------------------------
# 3. PCA: Steg 1 – Centrera datan
# -----------------------------
# Varför centrera?
# PCA kräver att datan har medelvärdet 0 i varje dimension.
# Formeln: X_centered = X - mean(X)
# - mean(X) beräknas per kolumn (per dimension)
# - sedan subtraheras medelvärdet från varje datapunkt

mean_vector = np.mean(all_embeddings, axis=0)  # medelvärde per dimension
centered = all_embeddings - mean_vector        # centrera varje datapunkt

# Print för att se vad som händer
print("\n--- Steg 1: Centrering ---")
print("Medelvärde per dimension (första 5):", mean_vector[:5])
print("Första datapunkt före centrering (första 5 dimensioner):", all_embeddings[0][:5])
print("Första datapunkt efter centrering (första 5 dimensioner):", centered[0][:5])

# -----------------------------
# 4. PCA: Steg 2 – Kovariansmatris
# -----------------------------
# Kovariansmatrisen visar hur dimensionerna samvarierar
# Formeln: C = (X_centered^T · X_centered) / (n-1)
cov_matrix = np.cov(centered.T)
print("\n--- Steg 2: Kovariansmatris ---")
print("Shape:", cov_matrix.shape)
print("Exempel på 3x3 block (första tre dimensionerna):")
print(cov_matrix[:3, :3])

# -----------------------------
# 5. PCA: Steg 3 – Egenvärden och egenvektorer
# -----------------------------
# Viktigt: använda np.linalg.eigh för symmetrisk kovariansmatris → alltid reella egenvärden
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

# Sortera egenvärden fallande
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

print("\n--- Steg 3: Egenvärden och egenvektorer ---")
print("Topp 5 egenvärden:", eig_vals[:5])
print("Första egenvektorns första 5 element:", eig_vecs[:5, 0])

# -----------------------------
# 6. PCA: Steg 4 – Projektion till 2D
# -----------------------------
top2_vecs = eig_vecs[:, :2]  # de två huvudkomponenterna
embeddings_2d = centered.dot(top2_vecs).real  # ta realdelen för att undvika ComplexWarning

print("\n--- Steg 4: Projektion ---")
print("Shape efter projektion (2D):", embeddings_2d.shape)
print("Första punkten i 2D:", embeddings_2d[0])

# -----------------------------
# 7. Visualisering med Matplotlib
# -----------------------------
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], c="blue", label="Meningar")
plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], c="red", marker="*", s=200, label="Query")

for i, text in enumerate(texts):
    plt.text(embeddings_2d[i, 0]+0.02, embeddings_2d[i, 1]+0.02, text, fontsize=9)
plt.text(embeddings_2d[-1, 0]+0.02, embeddings_2d[-1, 1]+0.02, "QUERY: " + query, fontsize=10, fontweight="bold", color="black")

plt.title("Visualisering av meningar med PCA (2D)")
plt.legend()
plt.show()

# -----------------------------
# 8. Extra: t-SNE
# -----------------------------
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# embeddings_2d_tsne = tsne.fit_transform(all_embeddings)

# plt.figure(figsize=(10, 7))
# plt.scatter(embeddings_2d_tsne[:-1, 0], embeddings_2d_tsne[:-1, 1], c="blue", label="Meningar")
# plt.scatter(embeddings_2d_tsne[-1, 0], embeddings_2d_tsne[-1, 1], c="red", marker="*", s=200, label="Query")

# for i, text in enumerate(texts):
#     plt.text(embeddings_2d_tsne[i, 0]+0.02, embeddings_2d_tsne[i, 1]+0.02, text, fontsize=9)
# plt.text(embeddings_2d_tsne[-1, 0]+0.02, embeddings_2d_tsne[-1, 1]+0.02, "QUERY", fontsize=12, fontweight="bold", color="red")

# plt.title("Visualisering av meningar med t-SNE (2D)")
# plt.legend()
# plt.show()
