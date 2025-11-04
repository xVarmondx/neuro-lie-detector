import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Krok 1: Wczytanie przygotowanych danych ---
try:
    df = pd.read_csv("features.csv")
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku 'features.csv'. Uruchom najpierw skrypt do ekstrakcji cech.")
    exit()

print("Wczytano zbiór danych z cechami. Oto jego podgląd:")
print(df.head())
print(f"\nLiczba wierszy w zbiorze: {len(df)}")
print("Rozkład warunków:")
print(df['warunek'].value_counts())

# --- Krok 2: Przygotowanie danych do uczenia maszynowego ---
df.dropna(inplace=True)

X = df[['amplituda_p300']]  # Cecha
y = df['warunek'].map({"Prawda": 0, "Kłamstwo": 1}) # Etykieta (cel)

# --- Krok 3: Podział na zbiór treningowy i testowy ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nPodzielono dane na {len(X_train)} próbek treningowych i {len(X_test)} próbek testowych.")

# --- Krok 4: Budowa i trening modelu ---
print("\nTrenowanie modelu...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Trening zakończony!")

# --- Krok 5: Ocena modelu ---
print("\n--- OCENA SKUTECZNOŚCI MODELU ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Dokładność modelu na zbiorze testowym: {accuracy * 100:.2f}%")

# Wizualizacja macierzy pomyłek
print("\nGenerowanie macierzy pomyłek...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Prawda", "Kłamstwo"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz Pomyłek")
plt.show()