import numpy as np

# =============================================
# 🎓 STEG 1: Skapa tre tokens (tänk ord)
# =============================================
tokens = ["Jag", "älskar", "kaffe"]

# Vi låtsas att embeddings är 3-dimensionella
# (normalt är de 768+ i riktiga modeller!)
X = np.array([
    [1.0, 0.5, 0.2],  # "Jag"
    [0.9, 1.0, 0.3],  # "älskar"
    [0.2, 0.1, 0.9]   # "kaffe"
])

print("Embeddings (X):")
print(X)
print("Dimensioner:", X.shape)
print()

# =============================================
# 🎓 STEG 2: Skapa viktsmatriser (W_Q, W_K, W_V)
# =============================================
# De lärs normalt under träning, men vi sätter manuella värden.
W_Q = np.array([
    [0.8, 0.1, 0.3],
    [0.2, 0.9, 0.5],
    [0.1, 0.4, 0.7]
])

W_K = np.array([
    [0.5, 0.2, 0.1],
    [0.1, 0.7, 0.3],
    [0.3, 0.4, 0.8]
])

W_V = np.array([
    [0.2, 0.6, 0.1],
    [0.8, 0.1, 0.3],
    [0.5, 0.4, 0.9]
])

# | Symbol | Namn  | Roll                                 |
# | :----- | :---- | :----------------------------------- |
# | **Q**  | Query | Frågar: ”vad söker jag efter?”       |
# | **K**  | Key   | Svarar: ”vad representerar jag?”     |
# | **V**  | Value | Innehållet/värdet som ska kombineras |

# =============================================
# 🎓 STEG 3: Beräkna Q, K, V
# =============================================
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q (Queries):\n", Q)
print("K (Keys):\n", K)
print("V (Values):\n", V)
print()


# =============================================
# 🎓 STEG 4: Beräkna likheter = QK^T
# =============================================
scores = Q @ K.T
print("Råa attention-scores (QK^T):")
print(scores)
print()

# =============================================
# 🎓 STEG 5: Skala med √d_k
# =============================================
d_k = K.shape[1]
scores_scaled = scores / np.sqrt(d_k)
print("Skalade scores (delat med sqrt(d_k)):")
print(scores_scaled)
print()

# =============================================
# 🎓 STEG 6: Softmax för att få vikter
# =============================================
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores_scaled)
print("Attention-vikter (efter softmax):")
print(attention_weights)
print()

# =============================================
# 🎓 STEG 7: Vikta Values med dessa vikter
# =============================================
output = attention_weights @ V
print("Output (sammanvägda representationer):")
print(output)
print()

# =============================================
# 🎓 STEG 8: Tolka resultatet
# =============================================
print("""
Varje rad i 'output' är en ny representation av varje token.
Den har nu "sett" hela meningen och blandat in kontext från andra ord.
Exempel:
- 'älskar' får information från 'Jag' och 'kaffe'
- 'kaffe' förstår sitt sammanhang (objekt till älskar)
Detta är kärnan i self-attention!
""")
