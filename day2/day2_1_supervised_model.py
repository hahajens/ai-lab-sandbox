# day2_explained.py
"""
Dag 2 — Textklassificering med tydliga förklaringar och stegvisa utskrifter.
Syfte: visa varje steg (data -> vektorisering -> träning -> prediktion -> utvärdering)
med enkla förklaringar (på svenska) och numeriska exempel.

Kör:
  python day2_explained.py
Förutsättningar (i aktiverad venv):
  pip install numpy pandas scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 0) Hjärtligt intro (printas så du läser innan beräkningar)
# ------------------------------------------------------
print("\n=== DAG 2: KLASSIFICERING — steg-för-steg med förklaringar ===\n")
print("Vi går igenom: data -> text->vektor -> träna modell -> utvärdera.")
print("Jag förklarar varje tal och visar formel där det behövs.\n")

# ------------------------------------------------------
# 1) DATA: ett litet exempel (du kan ersätta med din egen)
# ------------------------------------------------------
rows = [
    ("Jag kan inte logga in på min dator", "IT"),
    ("Får felmeddelande när jag sparar", "IT"),
    ("Hur ser jag mitt lönebesked?", "Payroll"),
    ("När får jag min lön denna månad?", "Payroll"),
    ("Hur ansöker jag om semester?", "HR"),
    ("Behöver ändra bankkonto för lön", "Payroll"),
    ("Min e-post fungerar inte", "IT"),
    ("Kan jag kombinera semester med tjänstledighet?", "HR"),
    ("Får jag ersättning för resa?", "Expenses"),
    ("Saknar behörighet i systemet", "IT"),
    ("Vart vänder jag mig för föräldraledighet?", "HR"),
    ("Fel 500 när jag besöker sidan", "IT"),
    ("Hur redovisar jag resekostnader?", "Expenses"),
    ("Hur ändrar jag adress i systemet?", "HR"),
    ("Lönebesked ser konstigt ut", "Payroll"),
    # några ambigua/utmanande exempel:
    ("Problem med bankkoppling och lön", "Payroll"),
    ("Datorn startar men blir svart skärm", "IT"),
    ("Vill veta hur sjuklön hanteras", "HR"),
    ("Får ingen avisering om lön", "Payroll"),
    ("Behöver hjälp att installera skrivardrivrutin", "IT"),
]
df = pd.DataFrame(rows, columns=["text", "label"])

print("1) Liten dataset skapad (exempel). Här är de första raderna:\n")
print(df.head(10).to_string(index=False))
print("\nAntal rader totalt:", len(df))
print("Klasser och antal per klass:")
print(df['label'].value_counts())
print("\nKommentar: detta är bara ett litet demo-dataset som vi använder för att förklara begrepp.\n")

# ------------------------------------------------------
# 2) Baseline: majority class
# ------------------------------------------------------
majority_label = df['label'].value_counts().idxmax()
baseline_acc = (df['label'] == majority_label).mean()
print("2) Baseline (majority) — enkel referens (gissa alltid den vanligaste klassen):")
print(f"   Majoritetsklass = '{majority_label}'. Ifall vi alltid gissar den, blir accuracy = {baseline_acc:.3f}")
print("   (Om en modell inte är bättre än det här, är den inte användbar.)\n")

# ------------------------------------------------------
# 3) Split: train/test
# ------------------------------------------------------
print("3) Dela data i tränings- och testset. Vi använder stratifiering för att behålla klassfördelning.")
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label'])
print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")
print("   Klassen i train (count):")
print(train_df['label'].value_counts())
print("   Klassen i test (count):")
print(test_df['label'].value_counts())
print()

# ------------------------------------------------------
# 4) Text -> siffror (vektorisering) — Count + TF-IDF (visa båda)
# ------------------------------------------------------
print("4) Text -> siffror (vektorisering). Vi visar både CountVectorizer (ordräkning)")
print("   och TF-IDF (vägd räkning som sänker " "vanliga ord")

# CountVectorizer (enkelt): räknar antal förekomster av ord/ngram
count_vect = CountVectorizer(ngram_range=(1,1), lowercase=True)
X_train_count = count_vect.fit_transform(train_df['text'])
X_test_count = count_vect.transform(test_df['text'])
print(f"   CountVectorizer: vocabulary size = {len(count_vect.vocabulary_)} ord")

# Visa hur en enskild text representeras som vektor (sparsamt) - visa ord med icke-nollvärden
example_idx = 0
example_text = train_df['text'].iloc[example_idx]
print("\n   Exempel: omvandling (Count) för text:", repr(example_text))
vec = X_train_count[example_idx].toarray().ravel()
nonzero_idx = np.where(vec > 0)[0]
print("   Ord och antal i denna text:")
for i in nonzero_idx:
    word = count_vect.get_feature_names_out()[i]
    count = vec[i]
    print(f"     '{word}': {int(count)}")

# TF-IDF (vanligare i textklassificering) — visar skillnad i värden
tfidf = TfidfVectorizer(ngram_range=(1,1), lowercase=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])
print("\n   TF-IDF bygger på ordfrekvens och 'idf' som sänker vikt för vanliga ord.")
print(f"   TF-IDF: vocabulary size = {len(tfidf.vocabulary_)} ord")
# Visa TF-IDF-värden för samma exempel
vec_tfidf = X_train_tfidf[example_idx].toarray().ravel()
nonzero_idx_tfidf = np.where(vec_tfidf > 0)[0]
print("   TF-IDF (ord : tf-idf-score) för samma text:")
for i in nonzero_idx_tfidf:
    word = tfidf.get_feature_names_out()[i]
    score = vec_tfidf[i]
    print(f"     '{word}': {score:.3f}")
print("\nKommentar: TF-IDF ger reella (flyttal) vikter, inte bara heltal; högre score = viktigare ord i dokumentet.\n")

# ------------------------------------------------------
# 5) Träna modell (Logistic Regression) på TF-IDF
# ------------------------------------------------------
print("5) Träna modell: Logistic Regression (en stabil, enkel baseline för text).")
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, train_df['label'])
print("   Modell tränad. (class_weight='balanced' hjälper vid obalans.)\n")

# ------------------------------------------------------
# 6) Prediktioner och sannolikheter på testset
# ------------------------------------------------------
print("6) Prediktioner på testset — visar varje exempel, sann etikett, prediction och sannolikheter.\n")
X_test = X_test_tfidf
y_test = test_df['label'].values
y_pred = model.predict(X_test)
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)
else:
    y_proba = None

# Visa per-exempel med förklaringar
classes = list(model.classes_)
for i, (text, true_label, pred_label) in enumerate(zip(test_df['text'], y_test, y_pred)):
    print(f"Ex {i+1}:")
    print(f"   Text: {text}")
    print(f"   Sann label: {true_label}")
    print(f"   Prediktion: {pred_label}")
    if y_proba is not None:
        probs = {classes[j]: float(y_proba[i, j]) for j in range(len(classes))}
        # sortera efter sannolikhet
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print("   Modellens sannolikheter (klass : sannolikhet):")
        for cls_name, p in sorted_probs:
            print(f"      {cls_name}: {p:.3f}")
    print("   (Kommentar: sannolikheten visar modellens 'tro' för varje klass.)\n")

# ------------------------------------------------------
# 7) Utvärdering: confusion matrix + metrics (manuellt och via sklearn)
# ------------------------------------------------------
print("7) Utvärdering på testset — confusion matrix och mått (precision, recall, f1).")
print("   Förklaring i korthet (enkelt språk):")
print("     - Precision: av de som modellen sade var klass X, hur många var rätt?")
print("     - Recall: av de som faktiskt var klass X, hur många hittade vi?")
print("     - F1: ett medelvärde mellan precision och recall (när du vill balansera båda).")
print()

labels_sorted = sorted(df['label'].unique())  # bestämd ordning för mat ris
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
print("Confusion matrix (rader = sanna klasser, kolumner = predikterade):")
print(cm_df.to_string())
print()

# Beräkna accuracy enkelt
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (andel rätt): {acc:.3f}  (Formel = antal_rätt / totalt)")

# Manuellt per-klass beräkning (TP, FP, FN, Precision, Recall, F1)
print("\nManuell uträkning per klass (visar formler och siffror):")
for idx, cls in enumerate(labels_sorted):
    TP = cm[idx, idx]
    FN = cm[idx, :].sum() - TP               # sanna men ej hittade (i rad)
    FP = cm[:, idx].sum() - TP               # predikterade men fel (i kolumn)
    TN = cm.sum() - (TP + FP + FN)
    # Skydda division med 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    print(f"  Klass '{cls}': TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"    Precision = TP/(TP+FP) = {TP}/{TP+FP} = {prec:.3f}")
    print(f"    Recall    = TP/(TP+FN) = {TP}/{TP+FN} = {rec:.3f}")
    print(f"    F1        = 2*(P*R)/(P+R) = {f1:.3f}\n")

# Också visa sklearn's snygga rapport
print("Sklearn classification_report (samma mått per klass):\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------
# 8) Plotta confusion matrix och spara bild (om du vill)
# ------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediktion")
plt.ylabel("Sann etikett")
plt.title("Confusion Matrix (testset)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix sparad som 'confusion_matrix.png' i din mapp.")
plt.show()

# ------------------------------------------------------
# 9) Felanalys: lista alla missar och varför modellen kunde tro så
# ------------------------------------------------------
print("\n8) FELANALYS — Vi visar vilka rader modellen missade och ger ledtrådar")
feature_names = np.array(tfidf.get_feature_names_out())
# Koef-matris: för varje klass finns en rad med viktningar för varje feature
coef = model.coef_   # form: (n_classes, n_features) — varje rad korresponderar med classes i model.classes_
class_to_idx = {c: i for i, c in enumerate(model.classes_)}

for i, (text, true_label, pred_label) in enumerate(zip(test_df['text'], y_test, y_pred)):
    if true_label != pred_label:
        print("----")
        print("Text:", text)
        print("True:", true_label, "| Pred:", pred_label)
        # Hitta vilka ord i texten bidrog mest till prediktionen
        x_vec = tfidf.transform([text])  # 1 x n_features
        x_indices = x_vec.nonzero()[1]   # index på aktiva features
        # för varje aktiv feature visa ord, tf-idf-score, och koefficient för predikterad klass
        print("Ord i texten (ord : tfidf-score) och coefficients (för predikterad klass):")
        infos = []
        for idx in x_indices:
            word = feature_names[idx]
            tfidf_score = x_vec[0, idx]
            coef_for_pred = coef[class_to_idx[pred_label], idx]
            infos.append((word, float(tfidf_score), float(coef_for_pred)))
        # Sortera efter (coef * tfidf) uppskattning av påverkan
        infos_sorted = sorted(infos, key=lambda t: t[2]*t[1], reverse=True)
        for word, score, c in infos_sorted[:8]:
            print(f"   {word:15s} | tfidf={score:.3f} | coef(pred_class)={c:.3f} | contrib≈{score*c:.3f}")
        print(" (Obs: positiv coef betyder ordet talar för predikterad klass; negativ talar emot.)")
        print()

# ------------------------------------------------------
# 10) Topp-vikter per klass — vilka ord talar mest för en klass?
# ------------------------------------------------------
print("9) Toppord (features) modellen använder för att känna igen varje klass:\n")
n_top = 10
for cls in model.classes_:
    idx = class_to_idx[cls]
    coefs = coef[idx]
    top_pos_idx = np.argsort(coefs)[-n_top:][::-1]
    top_neg_idx = np.argsort(coefs)[:n_top]
    print(f"Klass '{cls}' — topp positiva ord (talar för klassen):")
    for i_feat in top_pos_idx:
        print(f"   {feature_names[i_feat]:20s} | coef={coefs[i_feat]:.3f}")
    print()
print("\n (Slut på analys.)\n")

# Avslutningstips
print("Tips för nästa steg (om du vill förbättra modellen):")
print(" - Lägg till mer träningsdata (störst effekt).")
print(" - Använd n-gram (ngram_range=(1,2)) i TF-IDF så du fångar 'löne besked' etc.")
print(" - Testa MultinomialNB (bra baseline för text) och jämför.")
print(" - Gör felanalys: skapa regler för återkommande misstag (t ex 'lön' i text -> payroll).")
print("\nKlar! Kör gärna skriptet flera gånger efter du ändrat dataset eller parametrar.\n")
