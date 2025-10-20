# ===========================
# Dag 14 - Repetition & Mini-projekt
# ===========================

# ---------------------------
# Importera bibliotek
# ---------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------
# Skapa enkel dataset
# ---------------------------
# Features: timmar studerat, timmar sovit
# Label: klarade provet? 1 = Ja, 0 = Nej
X = np.array([
    [5, 6],  # elev 1
    [2, 8],  # elev 2
    [8, 5],  # elev 3
    [1, 4],  # elev 4
    [6, 7],  # elev 5
])
y = np.array([1, 0, 1, 0, 1])  # 1 = klarade, 0 = inte klarade

print("Features (X):\n", X)
print("Labels (y):\n", y)

# ---------------------------
# Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("\nTrain features:\n", X_train)
print("Train labels:\n", y_train)
print("\nTest features:\n", X_test)
print("Test labels:\n", y_test)

# ---------------------------
# Träna logistisk regression
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Modellparametrar
# ---------------------------
w = model.coef_[0]  # vikter
b = model.intercept_[0]  # bias
print("\nModellvikter (w):", w)
print("Modell-bias (b):", b)

# ---------------------------
# Odds ratio för tolkning
# ---------------------------
odds_ratio_feature1 = np.exp(w[0])
odds_ratio_feature2 = np.exp(w[1])
print("\nOdds ratio feature1 (timmar studerat):", odds_ratio_feature1)
print("Odds ratio feature2 (timmar sovit):", odds_ratio_feature2)

# ---------------------------
# Prediktion på testdata
# ---------------------------
y_pred = model.predict(X_test)
print("\nPrediktioner på testdata:", y_pred)

# ---------------------------
# Beräkna precision, recall, F1-score
# ---------------------------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# ---------------------------
# Extra pedagogik: confusion matrix
# ---------------------------
TP = np.sum((y_test==1) & (y_pred==1))
FP = np.sum((y_test==0) & (y_pred==1))
TN = np.sum((y_test==0) & (y_pred==0))
FN = np.sum((y_test==1) & (y_pred==0))

print("\nConfusion Matrix:")
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
