# ==============================
# DAG 13: F1-SCORE & FELANALYS
# ==============================

# Vi börjar med några exempeldata
# y_true = sanna labels
# y_pred = modellens förutsägelser

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# -----------------------------
# Steg 1: Definiera sanna värden och prediktioner
# -----------------------------
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])  # 1 = positiv, 0 = negativ
y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 0])

print("Sanna labels:   ", y_true)
print("Prediktioner:    ", y_pred)

# -----------------------------
# Steg 2: Skapa confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[0,0]

print("\nConfusion Matrix:\n", cm)
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): {TN}")

# -----------------------------
# Steg 3: Beräkna precision, recall och F1
# -----------------------------
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nBERÄKNING AV PRECISION, RECALL & F1")
print(f"Precision = TP / (TP + FP) = {TP} / ({TP}+{FP}) = {precision:.2f}")
print(f"Recall    = TP / (TP + FN) = {TP} / ({TP}+{FN}) = {recall:.2f}")
print(f"F1-score  = 2 * (P*R)/(P+R) = {f1:.2f}")

# -----------------------------
# Steg 4: Enkel felanalys
# -----------------------------
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        print(f"Felklassning på index {i}: Sann={y_true[i]}, Pred={y_pred[i]}")

# -----------------------------
# Steg 5: Kontroll med sklearn (praktisk metod)
# -----------------------------
precision_skl = precision_score(y_true, y_pred)
recall_skl = recall_score(y_true, y_pred)
f1_skl = f1_score(y_true, y_pred)

print("\nSKLEARN KONTROLL")
print(f"Precision (sklearn) = {precision_skl:.2f}")
print(f"Recall (sklearn)    = {recall_skl:.2f}")
print(f"F1-score (sklearn)  = {f1_skl:.2f}")
