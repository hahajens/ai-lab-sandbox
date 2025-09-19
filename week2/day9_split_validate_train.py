# =========================================
# DAG 9: Train/Validation/Test Split - PRAKTIK
# =========================================

import numpy as np
from sklearn.model_selection import train_test_split  # Funktion för att dela upp data

# =========================================
# Steg 0: Skapa syntetisk data
# =========================================
# Vi skapar 20 datapunkter med 2 features vardera, och labels 0 eller 1
# Features = information vi använder för att förutsäga label
# Label = facit (t.ex. äpple=0, banan=1)
np.random.seed(42)  # För att slumpen ska ge samma resultat varje gång

X = np.random.randn(20, 2)  # 20 datapunkter, 2 features vardera
y = np.random.randint(0, 2, 20)  # Labels: 0 eller 1

print("Full dataset (features) shape:", X.shape)
print("Full labels shape:", y.shape)
print("Exempel datapunkt:", X[0], "Label:", y[0])

# =========================================
# Steg 1: Split train + temp (val+test)
# =========================================
# Vi delar först data i 70% train och 30% temp
# stratify=y gör att klassfördelningen bevaras
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,      # 30% går till temp (sen delar vi det till val+test)
    random_state=42,    # För reproducerbarhet
    stratify=y          # Bevarar proportion av 0 och 1
)

print("\nTrain set shape:", X_train.shape, "Labels:", y_train)
print("Temp set shape (val+test):", X_temp.shape, "Labels:", y_temp)

# =========================================
# Steg 2: Split temp i validation + test
# =========================================
# Vi vill ha 15% val och 15% test (av totalt 20 datapunkter)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,      # Halva temp blir val, halva blir test
    random_state=42,
    stratify=y_temp
)

print("\nValidation set shape:", X_val.shape, "Labels:", y_val)
print("Test set shape:", X_test.shape, "Labels:", y_test)

# =========================================
# Steg 3: Kontrollera proportioner
# =========================================
print("\nAndel klasser i hela datasetet:")
print("Full dataset:", np.bincount(y))
print("Train:", np.bincount(y_train))
print("Validation:", np.bincount(y_val))
print("Test:", np.bincount(y_test))

# =========================================
# Steg 4: Visa några exempel
# =========================================
print("\nExempel på datapunkter i train set:")
for i in range(min(5, len(X_train))):
    print("Datapunkt:", X_train[i], "Label:", y_train[i])
