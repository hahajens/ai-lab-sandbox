import numpy as np
import matplotlib.pyplot as plt

# ==============================
# DAG 10: LOGISTIC REGRESSION - PEDAGOGISK LAB MED KOMMENTARER
# ==============================

# -----------------------------
# Steg 1: Sigmoid-funktion
# -----------------------------
# Denna funktion omvandlar ett värde z (råoutput från linjär kombination) till en sannolikhet mellan 0 och 1.
#
# Varför? I logistic regression vill vi att modellen ska ge ett sannolikhetsvärde för klass 1.
# z kan vara vilket reellt tal som helst, t.ex. -4 eller 3.5, och kan sträcka sig från minus oändlighet till plus oändlighet.
# Sigmoid-funktionen pressar dessa värden till intervallet [0,1]:
# - Om z är mycket negativt → σ(z) ≈ 0 → låg sannolikhet för klass 1
# - Om z är 0 → σ(z) = 0.5 → osäkerhet, lika stor chans för klass 0 eller 1
# - Om z är mycket positivt → σ(z) ≈ 1 → hög sannolikhet för klass 1
#
# Formeln:
# σ(z) = 1 / (1 + e^(-z))
# - np.exp(-z) = e upphöjt till -z, där e ≈ 2.718
# - 1 + np.exp(-z) = nämnaren som normaliserar värdet
# - 1 / (...) = vi får ett tal mellan 0 och 1
#
# I praktiken betyder detta att varje datapunkt får ett sannolikhetsvärde som sedan kan jämföras med en tröskel (t.ex. 0.5) för att bestämma predicerad klass.
#
# z = w*x + b där:
# w = vikt (hur mycket x påverkar z)
# x = feature (t.ex. antal timmar pluggat)
# b = bias (justerar kurvan upp/ner)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# Steg 2: Skapa enkel dataset
# -----------------------------
# x_values = antal timmar pluggat (feature)
# labels = sanna resultat, 0 = ej klar, 1 = klar (label)
x_values = np.array([0, 1, 2, 3, 4, 5])
labels = np.array([0, 0, 0, 1, 1, 1])

# -----------------------------
# Steg 3: Sätt initiala vikter och bias
# -----------------------------
# w = vikt för feature x, hur starkt x påverkar z
# b = bias, flyttar hela kurvan åt höger/ vänster
w = 1.5  # vikt
b = -4   # bias

# -----------------------------
# Steg 4: Beräkna linjär kombination z = w*x + b
# -----------------------------
# Förklarande kommentar:
# Här beräknar vi "råvärdet" z för varje datapunkt innan vi omvandlar det till sannolikhet.
# - z = w*x + b, en linjär kombination av feature och bias.
# - w (vikten) bestämmer hur mycket feature påverkar resultatet.
# - b (bias) flyttar kurvan upp eller ner för att justera modellen.
# - z kan vara vilket reellt tal som helst, t.ex. -2.3 eller 1.5.
# - Senare kommer sigmoid(z) omvandla detta till sannolikhet mellan 0 och 1.

# Linjär kombination = summan av viktade features + konstant (bias)

# Kan vara vilket tal som helst (negativt, positivt)

# I logistic regression används z som input till sigmoid-funktionen för att omvandla detta till en sannolikhet mellan 0 och 1
#
# Print-raden visar varje x och motsvarande z, och använder formatet :.2f för att skriva z med 2 decimaler, vilket gör det mer läsbart.
z = w * x_values + b
print("\n=== Linjär kombination (z) ===")
for i, val in enumerate(x_values):
    print(f"x={val} timmar → z = {z[i]:.2f}")

# -----------------------------
# Steg 5: Beräkna sannolikheter med sigmoid
# -----------------------------
# Förklarande kommentar:
# Här omvandlar vi de tidigare beräknade z-värdena (råoutput från linjär kombination) till sannolikheter.
# Sigmoid-funktionen tar varje z och pressar det till intervallet 0 till 1.
# - Om z är stort positivt → σ(z) ≈ 1 → mycket hög sannolikhet för klass 1.
# - Om z är stort negativt → σ(z) ≈ 0 → mycket låg sannolikhet för klass 1.
# - Om z ≈ 0 → σ(z) ≈ 0.5 → osäkerhet, ca 50% chans för klass 1.
#
# Denna sannolikhet är det som logistic regression faktiskt predicerar innan vi bestämmer en klass med tröskel.


probabilities = sigmoid(z)
print("\n=== Sannolikheter enligt sigmoid ===")
for i, p in enumerate(probabilities):
# i = index i arrayen, p = sannolikheten för denna datapunkt
# :.3f = avrundar till 3 decimaler för tydligare utskrift
    print(f"x={x_values[i]} → sannolikhet att klara = {p:.3f}")


# Kommentar:
# - Varje p-värde representerar sannolikheten att studenten klarar provet baserat på x och modellens w och b.
# - Dessa sannolikheter används sedan för att bestämma predicerad klass med en tröskel, t.ex. 0.5.

# -----------------------------
# Steg 6: Prediktera klass baserat på tröskel 0.5
# -----------------------------
# Om sannolikhet > 0.5 → klass 1, annars klass 0
predicted_class = (probabilities > 0.5).astype(int)
print("\n=== Predicerade klasser (tröskel 0.5) ===")
for i, c in enumerate(predicted_class):
    print(f"x={x_values[i]} → predicerad klass = {c} (sann label={labels[i]})")

# -----------------------------
# Steg 7: Visualisera datapunkter och sigmoid-kurva
# -----------------------------
# Blå linje = sigmoid-funktion
# Röd prick = sanna datapunkter (labels)
# Grön prick = predicerade sannolikheter
# Streckad linje = tröskel 0.5
x_range = np.linspace(-1, 6, 200)
z_curve = w * x_range + b
sigmoid_curve = sigmoid(z_curve)

plt.figure(figsize=(8,5))
plt.plot(x_range, sigmoid_curve, label="Sigmoid-funktion", color='blue')
plt.scatter(x_values, labels, color='red', label='Data (sanna labels)', zorder=5, s=100)
plt.scatter(x_values, probabilities, color='green', label='Predicerade sannolikheter', s=50)
plt.axhline(0.5, color='gray', linestyle='--', label='Tröskel 0.5')
plt.title("Logistic Regression - Pedagogisk visualisering")
plt.xlabel("Antal timmar pluggat")
plt.ylabel("Sannolikhet / Klass")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Steg 8: Enkel demonstration av gradient descent (en iteration)
# -----------------------------
# Målet: Minimera kostnadsfunktionen (log loss / cross-entropy)
# Log loss: J(w,b) = -(1/m) * Σ [y*log(y_hat) + (1-y)*log(1-y_hat)]
# Vi använder derivator (gradient) för att uppdatera w och b
# dz = skillnad mellan predicerad sannolikhet och sann label
m = len(x_values)
dz = probabilities - labels   # skillnad mellan pred och sann label
# dw = gradient för vikten, db = gradient för bias
dw = np.dot(dz, x_values) / m
db = np.sum(dz) / m
alpha = 0.1  # learning rate, styr hur stora steg vi tar

print("\n=== Gradient descent - en iteration ===")
print(f"dw = {dw:.3f}, db = {db:.3f}")
# Uppdatera vikter och bias
w_new = w - alpha * dw
b_new = b - alpha * db
print(f"Uppdaterade vikter: w={w_new:.3f}, b={b_new:.3f}")

# Kommentar:
# - Gradient descent tar små steg mot minimering av kostnaden.
# - dw och db visar lutningen för w och b.
# - Genom upprepade iterationer kan modellen lära sig bästa vikter för prediktion.
