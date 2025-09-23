import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# ======================================================
# DAG 12: Precision & Recall – pedagogiskt exempel
# ======================================================

# -----------------------------
# Steg 1: Skapa exempeldata
# -----------------------------
# Här skapar vi "faktiska labels" (y_true) och "modellens prediktioner" (y_pred)
# 1 = positiv (t.ex. spam)
# 0 = negativ (t.ex. inte spam)
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0])

# Printout: visa data
print("=====================================")
print("Faktiska labels (y_true) = de sanna etiketterna")
print(y_true)
print("\nPrediktioner från modellen (y_pred)")
print(y_pred)
print("=====================================")

# -----------------------------
# Steg 2: Beräkna confusion matrix
# -----------------------------
# Confusion matrix visar hur modellen presterar:
# TN = True Negative (modellen sa negativ, var negativ)
# FP = False Positive (modellen sa positiv, var negativ) <- fel
# FN = False Negative (modellen sa negativ, var positiv) <- fel
# TP = True Positive (modellen sa positiv, var positiv)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Printout: visa varje komponent
print("\n--- Confusion Matrix ---")
print(f"True Negative (TN): {tn} -> korrekt negativa")
print(f"False Positive (FP): {fp} -> felaktigt positiva (falsk alarm)")
print(f"False Negative (FN): {fn} -> felaktigt negativa (missade positiva)")
print(f"True Positive (TP): {tp} -> korrekt positiva")
print("------------------------")

# -----------------------------
# Steg 3: Beräkna Precision
# -----------------------------
# Precision: Av alla gånger modellen sa "positiv", hur många gånger hade den rätt?
# Formeln: Precision = TP / (TP + FP)
precision = tp / (tp + fp)

# Printout: visa formel och resultat
print("\n--- Precision ---")
print(f"Formel: Precision = TP / (TP + FP)")
print(f"Beräkning: {tp} / ({tp}+{fp}) = {precision:.2f}")
print("Tolkning: Av alla positiva prediktioner, hur stor andel var korrekta?")

# -----------------------------
# Steg 4: Beräkna Recall
# -----------------------------
# Recall: Av alla faktiska positiva exempel, hur många fångade modellen?
# Formeln: Recall = TP / (TP + FN)
recall = tp / (tp + fn)

# Printout: visa formel och resultat
print("\n--- Recall ---")
print(f"Formel: Recall = TP / (TP + FN)")
print(f"Beräkning: {tp} / ({tp}+{fn}) = {recall:.2f}")
print("Tolkning: Av alla faktiska positiva exempel, hur många hittade modellen?")
print("------------------------")

# -----------------------------
# Steg 5: Bekräfta med sklearn
# -----------------------------
precision_sklearn = precision_score(y_true, y_pred)
recall_sklearn = recall_score(y_true, y_pred)

print("\n--- Bekräftelse med sklearn ---")
print(f"Precision (sklearn) = {precision_sklearn:.2f}")
print(f"Recall (sklearn)    = {recall_sklearn:.2f}")
print("Dessa ska matcha våra manuella beräkningar.")

# -----------------------------
# Extra pedagogiskt steg: skriv ut vilka indexes som är TP, FP, FN, TN
# -----------------------------
print("\n--- Identifiera vilka exempel som är TP, FP, FN, TN ---")
for i in range(len(y_true)):
    if y_true[i]==1 and y_pred[i]==1:
        print(f"Index {i}: True Positive (rätt upptäckt)")
    elif y_true[i]==0 and y_pred[i]==1:
        print(f"Index {i}: False Positive (falsk alarm)")
    elif y_true[i]==1 and y_pred[i]==0:
        print(f"Index {i}: False Negative (missad)")
    elif y_true[i]==0 and y_pred[i]==0:
        print(f"Index {i}: True Negative (korrekt negativ)")

# ==============================
# Kort summering:
# Precision = hur säker är modellen när den säger "ja"?
# Recall = hur bra fångar modellen alla "ja" exempel?
# Det är en balans mellan att inte ge falska positiva och att inte missa positiva.
# ==============================
