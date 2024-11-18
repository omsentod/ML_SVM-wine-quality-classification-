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
    'kernel': ['rbf'],  # Kernel rbf seringkali memberikan hasil terbaik untuk dataset ini
}

# Stratified K-Fold untuk memastikan pembagian data yang seimbang
cv_splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV untuk mencari parameter terbaik
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=cv_splits, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Menampilkan hasil GridSearchCV SVM
print(f"\nBest parameters from GridSearchCV (SVM): {grid_search_svm.best_params_}")
svm_model = grid_search_svm.best_estimator_

# Prediksi dan evaluasi model SVM
y_pred_svm = svm_model.predict(X_test)

# Menampilkan hasil evaluasi
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nAkurasi Model SVM setelah tuning: {:.2f}%".format(accuracy_svm * 100))

# Precision, Recall, dan F1-Score untuk kelas 'Good' dan 'Bad'
precision_good = precision_score(y_test, y_pred_svm, pos_label='Good')
recall_good = recall_score(y_test, y_pred_svm, pos_label='Good')
f1_good = f1_score(y_test, y_pred_svm, pos_label='Good')

precision_bad = precision_score(y_test, y_pred_svm, pos_label='Bad')
recall_bad = recall_score(y_test, y_pred_svm, pos_label='Bad')
f1_bad = f1_score(y_test, y_pred_svm, pos_label='Bad')

print(f"\nPrecision (Good): {precision_good:.2f}")
print(f"Recall (Good): {recall_good:.2f}")
print(f"F1-Score (Good): {f1_good:.2f}")
print(f"Precision (Bad): {precision_bad:.2f}")
print(f"Recall (Bad): {recall_bad:.2f}")
print(f"F1-Score (Bad): {f1_bad:.2f}")

# Confusion Matrix untuk SVM
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=['Good', 'Bad'])
print("\nConfusion Matrix SVM:")
print(cm_svm)

# Visualisasi Confusion Matrix untuk SVM
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.title('Confusion Matrix SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Cross-validation untuk model SVM dengan best parameters
svm_scores = cross_val_score(SVC(C=grid_search_svm.best_params_['C'], 
                                    gamma=grid_search_svm.best_params_['gamma'], 
                                    kernel=grid_search_svm.best_params_['kernel']),
                                X_train, y_train, cv=cv_splits, scoring='accuracy')

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

# Fungsi untuk mengklasifikasikan wine berdasarkan fitur yang diberikan
def classify_wine(features):
    # Normalisasi fitur menggunakan scaler yang telah dilatih
    features_scaled = scaler.transform([features])  # Normalisasi data input
    # Reduksi dimensi dengan PCA
    features_pca = pca.transform(features_scaled)  # Mengurangi dimensi input sesuai dengan PCA yang sudah dilatih
    # Prediksi menggunakan model SVM
    prediction = svm_model.predict(features_pca)
    return prediction[0]  # Mengembalikan prediksi sebagai string

# Contoh input fitur untuk prediksi
example_red_wine = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]  # Fitur contoh Red Wine
example_white_wine = [6.0, 0.27, 0.0, 0.0, 0.038, 17.0, 60.0, 0.994, 3.15, 0.45, 2.8]  # Fitur contoh White Wine

# Prediksi untuk Red Wine dan White Wine
prediction_red_wine = classify_wine(example_red_wine)
prediction_white_wine = classify_wine(example_white_wine)

print(f"Prediksi untuk Red Wine dengan fitur {example_red_wine}: {prediction_red_wine}")
print(f"Prediksi untuk White Wine dengan fitur {example_white_wine}: {prediction_white_wine}")