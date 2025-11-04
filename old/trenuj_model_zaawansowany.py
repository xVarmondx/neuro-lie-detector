import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Wczytaj NOWY plik z cechami
df = pd.read_csv("features_rozbudowane.csv")
df.dropna(inplace=True)

# Definiujemy cechy (X) i etykiety (y)
feature_names = ["p300_mean_amp", "p300_peak_amp", "theta_power", "alpha_power", "beta_power"]
X = df[feature_names]
y = df['warunek'].map({"Prawda": 0, "Kłamstwo": 1})

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Używamy Lasu Losowego
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Ocena modelu
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- WYNIK ZAAWANSOWANEGO MODELU ---")
print(f"✅ Dokładność modelu (Random Forest): {accuracy * 100:.2f}%")

# Macierz pomyłek
disp = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test,
                                             display_labels=["Prawda", "Kłamstwo"], cmap=plt.cm.Blues)
plt.title("Macierz Pomyłek (Random Forest)")
plt.show()


# --- NOWA CZĘŚĆ: ANALIZA WAŻNOŚCI CECH ---
print("\n--- Analiza Ważności Cech ---")

# Pobieramy ważność cech z wytrenowanego modelu
importances = model.feature_importances_

# Tworzymy DataFrame dla lepszej wizualizacji
feature_importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Ważność': importances
}).sort_values(by='Ważność', ascending=False)

print(feature_importance_df)

# Tworzymy wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Cecha'], feature_importance_df['Ważność'], color='skyblue')
plt.xlabel('Ważność')
plt.title('Ważność Cech w Modelu')
plt.gca().invert_yaxis() # Wyświetlamy najważniejszą cechę na górze
plt.show()