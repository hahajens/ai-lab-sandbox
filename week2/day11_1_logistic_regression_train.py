# =========================================
# DAG 11: TRÄNA FÖRSTA KLASSIFICERAREN
# =========================================
# Målet: Bygga en enkel klassificerare med logistisk regression
# och förstå exakt vad som händer steg för steg.
#
# Klassificerare = en modell som förutspår om något tillhör klass 0 eller klass 1
# (t.ex. gillar glass = 1, gillar inte glass = 0)
# =========================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Steg 1: Skapa lite exempeldata
# -----------------------------
# Varje rad i X är en person
# Kolumn 1 = Temperatur ute (grader C)
# Kolumn 2 = Är det helg? (0=nej, 1=ja)

X = np.array([
    [30, 1],   # Varm dag + helg
    [25, 1],   # Ganska varm + helg
    [10, 0],   # Kallt + vardag
    [15, 0],   # Svalt + vardag
    [28, 0],   # Varmt + vardag
    [5, 0],    # Mycket kallt + vardag
    [32, 1],   # Väldigt varmt + helg
    [20, 1],   # Lagom varmt + helg
])

# y innehåller "facit" (labels)
# 1 = gillar glass, 0 = gillar inte glass
y = np.array([1, 1, 0, 0, 1, 0, 1, 1])

print("=== DATASET ===")
print("Feature-matris (X):")
print(X)
print("Facit (y):", y)
print("Varje rad i X hör ihop med ett värde i y\n")

# -----------------------------
# Steg 2: Dela upp i träning och test
# -----------------------------
# Träning: modellen får se dessa exempel och lära sig
# Test: vi gömmer undan några exempel för att kolla hur bra modellen lärde sig

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("=== TRÄNINGS- OCH TESTDATA ===")
print("Träningsdata (X_train):")
print(X_train)
print("Träningsfacit (y_train):", y_train)
print("\nTestdata (X_test):")
print(X_test)
print("Testfacit (y_test):", y_test)
print("\n")

# -----------------------------
# Steg 3: Skapa och träna modellen
# -----------------------------
# Vi väljer en logistisk regression-modell
model = LogisticRegression()

# Träna modellen på träningsdata
model.fit(X_train, y_train)

print("=== MODELL EFTER TRÄNING ===")
print("Vikter (w):", model.coef_)
print("Bias (b):", model.intercept_)
print("OBS: Vikterna visar hur viktiga features är för sannolikheten att gilla glass\n")

# -----------------------------
# Steg 4: Testa modellen
# -----------------------------
# Vi ber modellen gissa på X_test
y_pred = model.predict(X_test)              # Ger 0 eller 1 (klass)
y_proba = model.predict_proba(X_test)       # Ger sannolikheter (mellan 0 och 1)

print("=== PREDIKTIONER ===")
print("Faktiska facit (y_test):", y_test)
print("Modellens gissningar (y_pred):", y_pred)
print("Modellens sannolikheter (y_proba):")
print(y_proba)
print("\nVarje rad i sannolikheter är: [P(klass=0), P(klass=1)]\n")

# -----------------------------
# Steg 5: Utvärdera modellen
# -----------------------------
accuracy = model.score(X_test, y_test)
print("=== UTVÄRDERING ===")
print("Noggrannhet (accuracy):", accuracy)
print("Det betyder att modellen gissade rätt i", accuracy*100, "% av fallen")
