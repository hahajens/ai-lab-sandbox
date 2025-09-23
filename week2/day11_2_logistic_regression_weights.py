import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data (samma som tidigare)
X = np.array([
    [30, 1],
    [25, 1],
    [10, 0],
    [15, 0],
    [28, 0],
    [5, 0],
    [32, 1],
    [20, 1],
])
y = np.array([1, 1, 0, 0, 1, 0, 1, 1])

# Dela upp data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Träna modell
model = LogisticRegression()
model.fit(X_train, y_train)

# Visa vikter och bias
w = model.coef_[0]   # [w1, w2]
b = model.intercept_[0]

# =========================================
# TOLKA VIKTER MED ODDS RATIO
# =========================================

# Vi har tränat vår modell och fått vikterna:
# w[0] = vikt för temperatur
# w[1] = vikt för helg
# Bias b = modellens "startpunkt"
print("=== MODELLPARAMETRAR ===")
print("Vikt för temperatur (w1):", w[0])
print("Vikt för helg (w2):", w[1])
print("Bias (b):", b, "\n")

# -----------------------------
# Vad är log-odds?
# -----------------------------
# Modellen beräknar först något som kallas log-odds:
# logit(p) = ln(p / (1-p)) = w1*x1 + w2*x2 + b
# Där p är sannolikheten för klass 1 (t.ex. gilla glass)

# Exempel: Om w1=0.25 och x1=20 grader
# logit = 0.25 * 20 + w2*x2 + b
# Vi får alltså ett tal som kan vara >1 eller <0. 
# För att få en sannolikhet använder vi sigmoid: p = 1 / (1 + exp(-logit))

# -----------------------------
# Omvandla vikter till "odds ratio"
# -----------------------------
# För att göra det mer begripligt omvandlar vi vikterna från log-odds till odds ratio
# Formeln: odds_ratio = e^w

# Vad betyder detta?
# - odds_ratio_temp = np.exp(w[0])
#   → "Hur mycket multipliceras oddset för glass när temperaturen ökar med 1 grad?"
# - odds_ratio_helg = np.exp(w[1])
#   → "Hur mycket multipliceras oddset för glass om det är helg jämfört med vardag?"
# Omvandla till odds ratio (mer begripligt)
odds_ratio_temp = np.exp(w[0])
odds_ratio_helg = np.exp(w[1])

print("=== TOLKNING ===")
print(f"Temperatur: En grads ökning multiplicerar oddset för glass med {odds_ratio_temp:.2f} gånger")
print(f"Helg: Att det är helg multiplicerar oddset för glass med {odds_ratio_helg:.2f} gånger")

# Exempel: testa på en ny person
ny_person = np.array([[18, 0]])  # 18 grader, vardag
p = model.predict_proba(ny_person)[0][1]

print("\n=== EXEMPEL ===")
print("Ny person (18 grader, vardag): Sannolikhet att gilla glass =", round(p, 2))
