"""
=========================================
 DAG 8: INTRO TILL SUPERVISED LEARNING
=========================================

Supervised learning ("övervakad inlärning") betyder att vi tränar en modell
med exempel där vi redan vet facit (labels).

Målet: att modellen lär sig en funktion f(X) ≈ y
- X = features (siffror som beskriver något)
- y = labels (rätt svar, klassen vi vill förutsäga)

Exempel: [höjd, vikt] → människa eller hund

Varför splitta på train/test?
--------------------------------
- TRAIN DATA = används för att modellens parametrar ska läras
- TEST DATA  = används för att se hur bra modellen generaliserar till nya exempel
- Om vi bara testar på träningsdata får vi en falsk trygghet → risk för overfitting (memorering)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. Skapa en liten dataset
# -----------------------------
"""
Vi bygger en enkel dataset där varje rad (datapunkt) har två features:
[höjd i cm, vikt i kg]

Labels (facit) är:
1 = människa
0 = hund
"""

X = np.array([
    [170, 65],   # människa
    [180, 80],   # människa
    [160, 55],   # människa
    [30, 5],     # hund
    [25, 4],     # hund
    [35, 6],     # hund
])
y = np.array([1, 1, 1, 0, 0, 0])

print("\n=== DATASET ===")
print("Features (X):\n", X)
print("Labels (y):", y)
print("Förklaring: X är en matris (6x2) med höjd och vikt. y är en vektor med klasser.")

# -----------------------------
# 2. Splitta data i train/test
# -----------------------------
"""
Vi delar upp datasetet i två delar:
- 70% TRAIN: används för att lära modellen
- 30% TEST : används för att utvärdera modellen på nya exempel

Formeln här är inget avancerat, men tänk såhär:
Totalt antal datapunkter = N
Antal till träning = 0.7 * N
Antal till test    = 0.3 * N
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n=== TRAIN/TEST SPLIT ===")
print("Train features:\n", X_train)
print("Train labels:", y_train)
print("Test features:\n", X_test)
print("Test labels:", y_test)

# -----------------------------
# 3. Skapa och träna modellen
# -----------------------------
"""
Vi använder Logistic Regression (mer teori på dag 10).
Just nu räcker det att veta att modellen försöker hitta en gräns
som skiljer klass 0 från klass 1.

Träning = modellen justerar sina parametrar så att prediktionen
blir så nära labels som möjligt på TRAIN data.
"""

model = LogisticRegression()
model.fit(X_train, y_train)

print("\n=== MODELL TRÄNAD ===")
print("Modellen har nu 'lärt sig' att skilja hundar från människor baserat på höjd och vikt.")

# -----------------------------
# 4. Utvärdera modellen
# -----------------------------
"""
Vi mäter 'accuracy' (noggrannhet) = (antal rätt gissningar) / (antal testexempel)

Formel:
Accuracy = (rätt klassificeringar) / (totalt antal test-exempel)
"""

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\n=== UTVÄRDERING ===")
print(f"Noggrannhet på TRAIN data: {train_score:.2f}")
print(f"Noggrannhet på TEST data : {test_score:.2f}")
print("OBS: Om train-score >> test-score så är det risk för overfitting.")

# -----------------------------
# 5. Testa på nya exempel
# -----------------------------
"""
Nu testar vi modellen på helt nya punkter som den inte sett:
[40 cm, 7 kg]   → borde vara hund (låg vikt + kort)
[175 cm, 70 kg] → borde vara människa (stor + tung)

Vi använder metoden predict().
"""

# new_data = np.array([[40, 7], [175, 70]])

new_data = np.array([
    [40, 7],     # typisk hund
    [175, 70],   # typisk människa
    [90, 25]     # mycket liten människa (dvärgexempel)
])

new_pred = model.predict(new_data)
new_prob = model.predict_proba(new_data)  # sannolikheter för varje klass


print("\n=== NYA EXEMPEL ===")
for inp, pred, prob in zip(new_data, new_pred, new_prob):
    label = "människa" if pred == 1 else "hund"
    print(f"Input {inp} → Modellens prediktion: {label}")
    print(f"   Sannolikhet hund (klass 0): {prob[0]:.2f}")
    print(f"   Sannolikhet människa (klass 1): {prob[1]:.2f}")


# -----------------------------
# 6. Visualisering av beslutsgräns
# -----------------------------
"""
vi ritar upp planet (höjd vs vikt) och färglägger områden där modellen
anser att det är människa respektive hund.
Sedan placerar vi in våra testpunkter ovanpå.
"""


import matplotlib.pyplot as plt

# Facit för våra nya punkter (0 = hund, 1 = människa)
true_labels = np.array([0, 1, 1])

# Skapa rutnät av punkter
xx, yy = np.meshgrid(
    np.linspace(20, 200, 200),
    np.linspace(2, 100, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points).reshape(xx.shape)

# Rita bakgrund (områdena)
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# Rita träningsdata
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", label="Train data")

# Rita nya punkter (testexempel) med annotationer
for inp, pred, prob, true in zip(new_data, new_pred, new_prob, true_labels):
    color = "blue" if pred == 1 else "red"
    plt.scatter(inp[0], inp[1], marker="*", s=200, color=color, edgecolors="k")
    
    # Gör etiketttext
    pred_label = "människa" if pred == 1 else "hund"
    true_label = "människa" if true == 1 else "hund"
    text = (f"X={inp}\n"
            f"Pred: {pred_label}\n"
            f"True: {true_label}\n"
            f"P(människa)={prob[1]:.2f}")
    
    plt.text(inp[0]+2, inp[1]+2, text, fontsize=9,
             bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

plt.xlabel("Höjd (cm)")
plt.ylabel("Vikt (kg)")
plt.title("Beslutsgräns: hund (röd) vs människa (blå)")
plt.legend()
plt.show()
