# ===============================================
# DAG 15 – Introduktion till LLM: Tokens & Transformer
# ===============================================
# Vi ska se hur text omvandlas till tokens och hur en LLM förstår det.
# Vi använder Hugging Face (transformers) för att demonstrera.

from transformers import AutoTokenizer, AutoModel
import torch

# -----------------------------
# Steg 1: Ladda en tokenizer och modell
# -----------------------------
# Vi väljer en liten modell (distilbert-base-uncased)
# Den är snabb men bygger på samma principer som GPT-modeller.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# -----------------------------
# Steg 2: Tokenisera en mening
# -----------------------------
text = "ChatGPT is amazing and understands language."

# Tokenisera = omvandla text till token-ID:n (heltal)
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n--- TOKENISERING ---")
print("Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Antal tokens:", len(tokens))

# -----------------------------
# Steg 3: Skicka tokens till modellen
# -----------------------------
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# outputs innehåller dolda representationer (embeddings)
# Shape (batch_size, sequence_length, hidden_size)
print("\n--- MODELLUTDATA ---")
print("Output shape:", outputs.last_hidden_state.shape)

# -----------------------------
# Steg 4: Titta på embedding för ett token
# -----------------------------
# Varje token får en vektor (embedding)
first_token_embedding = outputs.last_hidden_state[0, 0, :]
print("\n--- EMBEDDING-EXEMPEL ---")
print("Dimension på embedding:", first_token_embedding.shape)
print("Första 10 värdena i vektorn:\n", first_token_embedding[:10])

# -----------------------------
# Pedagogisk reflektion
# -----------------------------
# Varje ord (token) → heltal → vektor
# Modellen lär sig relationer mellan vektorerna.
# Self-attention gör att modellen kan "se" hela meningen samtidigt
# och avgöra vilka ord som påverkar varandra mest.
