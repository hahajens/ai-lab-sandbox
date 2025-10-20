import numpy as np

# =====================================================
# STEG 1: Embeddings X
# =====================================================
# Teori: 
# Vi representerar varje token som en vektor (embedding).
# Dimensionen på vektorn = antal features per token (t.ex. 3 här).
# Varje rad = ett token, varje kolumn = feature.
tokens = ["Jag", "älskar", "kaffe"]
X = np.array([
    [1.0, 0.5, 0.2],  # Token "Jag"
    [0.9, 1.0, 0.3],  # Token "älskar"
    [0.2, 0.1, 0.9]   # Token "kaffe"
])

print("Embeddings X (rad=token, kolumn=dimension):")
print(X)
print()

# =====================================================
# STEG 2: Viktsmatriser W_Q, W_K, W_V
# =====================================================
# Teori:
# Self-attention använder tre matriser för att skapa:
# Q (Query) = Vad varje token “frågar efter”
# K (Key)   = Hur token representeras
# V (Value) = Innehållet som ska vägas ihop
# Q = X @ W_Q, K = X @ W_K, V = X @ W_V

W_Q = np.array([[0.8, 0.1, 0.3], [0.2, 0.9, 0.5], [0.1, 0.4, 0.7]])
W_K = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.3], [0.3, 0.4, 0.8]])
W_V = np.array([[0.2, 0.6, 0.1], [0.8, 0.1, 0.3], [0.5, 0.4, 0.9]])

print("W_Q, W_K, W_V:")
print("W_Q:\n", W_Q)
print("W_K:\n", W_K)
print("W_V:\n", W_V)
print()

# =====================================================
# STEG 3: Beräkna Q, K, V med detaljerad förklaring
# =====================================================
def detailed_matmul(A, B, labelA, labelB):
    """
    Matris-multiplikation med full steg-för-steg print:
    result[i,j] = sum_k(A[i,k]*B[k,j])
    Teori:
      - Q = X @ W_Q
      - K = X @ W_K
      - V = X @ W_V
    """
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    assert cols_A == rows_B, "Dimension mismatch!"
    result = np.zeros((rows_A, cols_B))
    
    print(f"--- {labelA} @ {labelB} ---")
    for i in range(rows_A):  # Token
        for j in range(cols_B):  # Output-dimension
            sum_products = 0
            product_details = []
            for k in range(cols_A):
                prod = A[i,k] * B[k,j]
                sum_products += prod
                product_details.append(f"{A[i,k]}*{B[k,j]}={prod:.3f}")
            result[i,j] = sum_products
            print(f"{labelA} rad {i} × {labelB} kol {j}: " +
                  " + ".join(product_details) +
                  f" = {sum_products:.3f}")
    print()
    return result

Q = detailed_matmul(X, W_Q, "X", "W_Q")  # Query
K = detailed_matmul(X, W_K, "X", "W_K")  # Key
V = detailed_matmul(X, W_V, "X", "V")    # Value

# =====================================================
# STEG 4: Beräkna QK^T
# =====================================================
def detailed_QKT(Q, K):
    """
    Beräkna dot-product mellan varje Query och varje Key.
    Teori:
      - score[i,j] = Q[i] · K[j] = sum_k(Q[i,k]*K[j,k])
      - Mäter hur mycket token i “passar med” token j
    """
    rows_Q, cols_Q = Q.shape
    rows_K, cols_K = K.shape
    result = np.zeros((rows_Q, rows_K))
    
    print("--- Q @ K^T ---")
    for i in range(rows_Q):
        for j in range(rows_K):
            sum_products = 0
            product_details = []
            for k in range(cols_Q):
                prod = Q[i,k] * K[j,k]
                sum_products += prod
                product_details.append(f"{Q[i,k]}*{K[j,k]}={prod:.3f}")
            result[i,j] = sum_products
            print(f"Q rad {i} · K rad {j}: " + " + ".join(product_details) +
                  f" = {sum_products:.3f}")
    print()
    return result

scores = detailed_QKT(Q, K)

# =====================================================
# STEG 5: Skala scores med sqrt(d_k)
# =====================================================
# Teori:
# Q, K och V är matriser där varje rad representerar ett token.

# Varje token-vektor har d_k dimensioner (dvs längden på vektorn).

# När vi beräknar attention:

# Om vektorerna har stora dimensioner, blir dot-produkten (Q·K) ofta väldigt stor.

# Problem: När vi skickar dessa stora scores in i softmax, blir exponenterna väldigt stora → softmax blir “för skarp” → en token får nästan all vikt.

# Lösning: vi delar med 
#  (kvadratroten av vektorns dimension):
# Vi dividerar med sqrt(d_k) för att förhindra att softmax blir för “skarp”
d_k = K.shape[1]
scores_scaled = np.zeros_like(scores)

print("--- Skala scores med sqrt(d_k) ---")
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        scaled = scores[i,j] / np.sqrt(d_k)
        scores_scaled[i,j] = scaled
        print(f"Score[{i},{j}] = {scores[i,j]:.3f} / sqrt({d_k}) = {scaled:.3f}")
print()

# =====================================================
# STEG 6: Softmax radvis
# =====================================================
def detailed_softmax(scores):
    """
    Softmax konverterar varje rad av scores till sannolikheter (attention weights)
    Teori:
      - softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
      - Radvis summerar vikterna till 1
    """
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    print("--- Softmax-vikter ---")
    for i in range(softmax_scores.shape[0]):
        print(f"Token {i} softmax-vikter = {softmax_scores[i]}")
        print(f"Förklaring: exp(scores - max) = {exp_scores[i]}, summera = {exp_scores[i].sum():.3f}")
    print()
    return softmax_scores

attention_weights = detailed_softmax(scores_scaled)

# =====================================================
# STEG 7: Vikta V med attention-vikter
# =====================================================
def apply_attention(attn_weights, V):
    """
    Slutgiltig self-attention:
      output[i] = sum_j(attn_weight[i,j] * V[j])
    Teori:
      - Varje token får ny representation med kontext från alla andra tokens
    """
    output = np.zeros_like(V)
    print("--- Vikta Values med attention-vikter ---")
    for i in range(attn_weights.shape[0]):
        row_output = np.zeros(V.shape[1])
        for j in range(attn_weights.shape[1]):
            contrib = attn_weights[i,j] * V[j]
            row_output += contrib
            print(f"Token {i}: + ({attn_weights[i,j]:.3f} * V[{j}]) = {contrib}")
        output[i] = row_output
        print(f"Resultat för token {i}: {output[i]}")
    print()
    return output

output = apply_attention(attention_weights, V)

print("✅ Slutgiltigt output (nya token-representationer med kontext):")
print(output)
