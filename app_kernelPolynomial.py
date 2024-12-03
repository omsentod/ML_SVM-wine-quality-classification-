import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
red_wine_path = 'winequality-red.csv'
white_wine_path = 'winequality-white.csv'

red_wine = pd.read_csv(red_wine_path, delimiter=';')
white_wine = pd.read_csv(white_wine_path, delimiter=';')

# Menambahkan kolom type untuk masing-masing wine
red_wine['type'] = 'Red'
white_wine['type'] = 'White'
data = pd.concat([red_wine, white_wine], axis=0)

# Menghapus outlier menggunakan IQR
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
            'density', 'pH', 'sulphates', 'alcohol']

Q1 = data[features].quantile(0.25)
Q3 = data[features].quantile(0.75)
IQR = Q3 - Q1

# Menghapus baris dengan nilai di luar bound IQR
data_cleaned = data[~((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Memilih fitur untuk model
X = data_cleaned[features]
y = data_cleaned['quality']

# Mengubah kualitas menjadi kategori "Good" atau "Bad"
y = y.apply(lambda q: 'Good' if q >= 6 else 'Bad')

# Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mengurangi dimensi fitur menggunakan PCA
pca = PCA(n_components=0.95)  # Mengambil komponen yang mencakup 95% varians
X_pca = pca.fit_transform(X_scaled)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

# Parameter Grid untuk SVM yang lebih efisien
param_grid_svm = {
    'C': [1, 10, 100],  # Mencoba nilai C yang lebih besar untuk meningkatkan akurasi
    'gamma': [0.01, 0.1, 'scale'],  # Menggunakan 'scale' lebih banyak digunakan untuk nilai gamma
    'kernel': ['poly'],  # Kernel rbf seringkali memberikan hasil terbaik untuk dataset ini
}

# Stratified K-Fold untuk memastikan pembagian data yang seimbang
cv_splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV untuk mencari parameter terbaik
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=cv_splits, scoring='accuracy', verbose=0, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Menampilkan hasil GridSearchCV
print("\n===== Hasil GridSearchCV =====")
print(f"Parameter untuk SVM: {grid_search_svm.best_params_}")

# Prediksi dan evaluasi model SVM
svm_model = grid_search_svm.best_estimator_
y_pred_svm = svm_model.predict(X_test)

# Akurasi model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_percent = accuracy_svm * 100
print("\n===== Akurasi Model =====")
print(f"Akurasi Model SVM setelah tuning: {accuracy_percent:.2f}%")

# Evaluasi precision, recall, dan f1-score untuk tiap kelas
precision_good = precision_score(y_test, y_pred_svm, pos_label='Good')
recall_good = recall_score(y_test, y_pred_svm, pos_label='Good')
f1_good = f1_score(y_test, y_pred_svm, pos_label='Good')

precision_bad = precision_score(y_test, y_pred_svm, pos_label='Bad')
recall_bad = recall_score(y_test, y_pred_svm, pos_label='Bad')
f1_bad = f1_score(y_test, y_pred_svm, pos_label='Bad')

print("\n===== Evaluasi Performa (dalam persen) =====")
print(f"Good: Precision: {precision_good * 100:.2f}%, Recall: {recall_good * 100:.2f}%, F1-Score: {f1_good * 100:.2f}%")
print(f"Bad : Precision: {precision_bad * 100:.2f}%, Recall: {recall_bad * 100:.2f}%, F1-Score: {f1_bad * 100:.2f}%")

# Menampilkan laporan klasifikasi
classification_report_dict = classification_report(y_test, y_pred_svm, zero_division=0, output_dict=True)
classification_report_df = pd.DataFrame(classification_report_dict).transpose()
classification_report_df[['precision', 'recall', 'f1-score']] = classification_report_df[['precision', 'recall', 'f1-score']] * 100
print("\n===== Laporan Klasifikasi (dalam persen) =====")
print(classification_report_df[['precision', 'recall', 'f1-score']].round(2))

# Menampilkan Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=['Good', 'Bad'])
print("\n===== Confusion Matrix =====")
print(cm_svm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.title('Confusion Matrix SVM')
plt.xlabel('Prediksi')
plt.ylabel('Kebenaran')
plt.show()

# Grafik 1: Quality vs Residual Sugar
plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='quality', y='residual sugar', errorbar='sd', capsize=0.2)
plt.title('Quality vs Residual Sugar')
plt.xlabel('Quality')
plt.ylabel('Residual Sugar')

plt.show()

# Grafik 2: Quality vs Free Sulfur Dioxide
plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='quality', y='free sulfur dioxide', errorbar='sd', capsize=0.2)
plt.title('Quality vs Free Sulfur Dioxide')
plt.xlabel('Quality')
plt.ylabel('Free Sulfur Dioxide')
plt.show()

# Grafik 3: Quality vs Citric Acid
plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='quality', y='citric acid', errorbar='sd', capsize=0.2)
plt.title('Quality vs Citric Acid')
plt.xlabel('Quality')
plt.ylabel('Citric Acid')
plt.show()

def classify_wine_from_input():
    print("\nPrediksi kualitas wine:")
    print("Masukkan fitur dalam urutan berikut, dipisahkan dengan koma:")
    print("[fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol]")
    print("Ketik 'stop' kapan saja untuk keluar.")

    # Meminta pengguna memilih jenis wine
    while True:
        wine_type = input("Pilih jenis wine (Red/White): ").strip().lower()
        if wine_type == 'stop':
            print("Proses dihentikan.")
            return
        if wine_type in ['red', 'white']:
            break
        else:
            print("Harap masukkan 'Red' atau 'White'.")

    # Memasukkan semua fitur dalam satu baris dengan pemisah koma
    while True:
        try:
            input_line = input("Masukkan nilai fitur (dipisahkan dengan koma): ")
            if input_line.strip().lower() == 'stop':
                print("Proses dihentikan.")
                return
            input_features = [float(value) for value in input_line.split(',')]
            if len(input_features) != 11:
                raise ValueError("Harap masukkan 11 nilai.")
            break
        except ValueError as e:
            print(f"Error: {e}. Harap masukkan 11 nilai numerik yang valid, dipisahkan dengan koma.")

    # Normalisasi data input menggunakan scaler yang telah dilatih
    features_scaled = scaler.transform([input_features])  # Normalisasi data input
    # Reduksi dimensi dengan PCA
    features_pca = pca.transform(features_scaled)  # Mengurangi dimensi input sesuai PCA yang sudah dilatih
    # Prediksi menggunakan model SVM
    prediction = svm_model.predict(features_pca)
    return prediction[0]

# Memanggil fungsi untuk input manual
print("\n===== Prediksi Wine Berdasarkan Input Manual =====")
prediction_manual = classify_wine_from_input()
if prediction_manual:
    print(f"Prediksi untuk wine berdasarkan input: {prediction_manual}")