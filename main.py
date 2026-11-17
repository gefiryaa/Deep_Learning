from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

# ============================================================
# 2. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "BlaBla.xlsx"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH}\n"
        "Pastikan file BlaBla.xlsx berada satu folder dengan file Python."
    )

df = pd.read_excel(DATASET_PATH)

print("=" * 60)
print("DATASET")
print("=" * 60)
print("Ukuran dataset:", df.shape)
print("\nLima data teratas:")
print(df.head())
print("\nInformasi dataset:")
print(df.info())


# ============================================================
# 3. PISAHKAN FITUR DAN LABEL
# Input  : UMUR_TAHUN, B sampai M
# Output : N
# ============================================================

required_columns = ["UMUR_TAHUN"] + list("BCDEFGHIJKLM") + ["N"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")

X = df[["UMUR_TAHUN"] + list("BCDEFGHIJKLM")].copy()
y = df["N"].copy()

print("\nKolom input:", X.columns.tolist())
print("Kolom output: N")


# ============================================================
# 4. ENCODING UMUR
# 1 = umur <= 20
# 2 = umur 21-30
# 3 = umur 31-40
# 4 = umur 41-50
# 5 = umur > 50
# ============================================================

X["UMUR_TAHUN"] = pd.to_numeric(X["UMUR_TAHUN"], errors="coerce")

invalid_age_count = int(X["UMUR_TAHUN"].isna().sum())
if invalid_age_count > 0:
    median_age = X["UMUR_TAHUN"].median()
    X["UMUR_TAHUN"] = X["UMUR_TAHUN"].fillna(median_age)
    print(
        f"\nDitemukan {invalid_age_count} nilai umur yang bukan angka; "
        f"diisi menggunakan median umur ({median_age})."
    )

# Pastikan kolom fitur lainnya numerik.
for col in list("BCDEFGHIJKLM"):
    X[col] = pd.to_numeric(X[col], errors="coerce")

# ============================================================
# 5. PREPROCESSING DATA
# a. Cek missing value
# b. Cek outlier menggunakan IQR pada umur
# ============================================================

print("\nMissing value sebelum penanganan:")
print(X.isnull().sum())

# Mengisi missing value numerik dengan median masing-masing kolom.
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

print("\nMissing value setelah penanganan:")
print(X.isnull().sum())

# Pemeriksaan outlier pada umur menggunakan IQR.
Q1 = X["UMUR_TAHUN"].quantile(0.25)
Q3 = X["UMUR_TAHUN"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (
    (X["UMUR_TAHUN"] < lower_bound) |
    (X["UMUR_TAHUN"] > upper_bound)
)
outlier_count = int(outlier_mask.sum())

print("\nPemeriksaan outlier umur (metode IQR):")
print(f"Batas bawah : {lower_bound:.2f}")
print(f"Batas atas  : {upper_bound:.2f}")
print(f"Jumlah outlier: {outlier_count}")

# Capping agar baris data tidak perlu dihapus.
X["UMUR_TAHUN"] = X["UMUR_TAHUN"].clip(
    lower=lower_bound,
    upper=upper_bound
)

# Encoding umur sesuai rentang pada modul.
X["UMUR_TAHUN"] = pd.cut(
    X["UMUR_TAHUN"],
    bins=[-np.inf, 20, 30, 40, 50, np.inf],
    labels=[1, 2, 3, 4, 5],
    right=True
).astype(int)

# Pastikan target berupa angka biner 0 dan 1.
y = pd.to_numeric(y, errors="coerce")
if y.isna().any():
    y = y.fillna(y.mode().iloc[0])

unique_target = sorted(y.unique().tolist())
if not set(unique_target).issubset({0, 1}):
    raise ValueError(
        f"Target N harus berupa klasifikasi biner 0/1. "
        f"Nilai yang ditemukan: {unique_target}"
    )
y = y.astype(int)

print("\nDistribusi label N:")
print(y.value_counts().sort_index())


# ============================================================
# 6. SPLIT DATA
# Data training 80% dan testing 20%
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", X_train.shape[0])
print("Jumlah data testing :", X_test.shape[0])

# Standardisasi fitur berdasarkan data training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 7. BANGUN MODEL KLASIFIKASI ANN
# Arsitektur mengikuti modul: Dense(16) -> Dense(8) -> Dense(1)
# ============================================================

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

print("\nRingkasan model:")
model.summary()


# ============================================================
# 8. COMPILE DAN TRAIN
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nMulai training model...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# 9. EVALUASI
# ============================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)
print(f"Loss              : {loss:.4f}")
print(f"Akurasi data uji  : {accuracy * 100:.2f}%")


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

y_probability = model.predict(X_test, verbose=0)
y_pred = (y_probability >= 0.5).astype(int).ravel()

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])


# ============================================================
# 11. TAMPILKAN CONFUSION MATRIX
# ============================================================


plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[0, 1],
    yticklabels=[0, 1]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

# Simpan confusion matrix
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================================================
# 12. LAPORAN KLASIFIKASI
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )
)

# Grafik akurasi dan loss selama training sebagai informasi tambahan.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Training Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()

# Simpan grafik training
plt.savefig(
    RESULT_DIR / "training_history.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nProgram Deep Learning selesai dijalankan.")
print(f"Hasil visualisasi tersimpan di: {RESULT_DIR}")
