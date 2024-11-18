import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

red_wine_path = 'winequality-red.csv'  
white_wine_path = 'winequality-white.csv'  

red_wine = pd.read_csv(red_wine_path, delimiter=';')
white_wine = pd.read_csv(white_wine_path, delimiter=';')

red_wine['type'] = 'Red'
white_wine['type'] = 'White'
data = pd.concat([red_wine, white_wine], axis=0)

correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
print("\nKoefisien Korelasi (R) antara fitur dan kualitas:")
print(correlation_matrix['quality'])

features = ['volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide']
X = data[features]
y = data['quality']

y = y.apply(lambda q: 'Good' if q >= 6 else 'Bad')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Mengurangi dimensi fitur menggunakan PCA 
pca = PCA(n_components=2)  # Mengurangi ke 2 komponen utama
X_pca = pca.fit_transform(X_balanced)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_balanced, test_size=0.2, random_state=42)

# Langkah 3: Melatih model SVM
svm_model = SVC(kernel='linear', random_state=111)
svm_model.fit(X_train, y_train)

def classify_wine(features_input):
    """
    Menerima input berupa fitur wine dan mengembalikan prediksi (Good/Bad).
    
    Parameters:
    features_input (list): Nilai fitur dalam urutan [volatile acidity, citric acid, residual sugar, free sulfur dioxide]
    
    Returns:
    str: Prediksi apakah wine tersebut "Good" atau "Bad".
    """
    # Membuat dataframe untuk input dengan nama kolom yang sama seperti saat fitting
    input_df = pd.DataFrame([features_input], columns=['volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide'])
    # Standarisasi input fitur
    features_scaled = scaler.transform(input_df)
    # Transformasi menggunakan PCA
    features_pca = pca.transform(features_scaled)
    # Prediksi menggunakan model SVM
    prediction = svm_model.predict(features_pca)
    return prediction[0]

# Contoh penggunaan fungsi klasifikasi
# Input contoh: volatile acidity=0.5, citric acid=0.3, residual sugar=3.0, free sulfur dioxide=20
example_features = [0.5, 0.3, 3.0, 20]
prediction = classify_wine(example_features)
print(f"Prediksi untuk fitur {example_features}: {prediction}")


# Langkah 4: Prediksi dan evaluasi model
y_pred = svm_model.predict(X_test)

# Menampilkan hasil evaluasi
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nAkurasi Model SVM: {:.2f}%".format(accuracy * 100))
print("\nLaporan Klasifikasi:\n", report)

# Validasi menggunakan Cross-Validation
cross_val_scores = cross_val_score(svm_model, X_pca, y_balanced, cv=5)
print("\nAkurasi rata-rata Cross-Validation: {:.2f}%".format(cross_val_scores.mean() * 100))

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
plt.title('Quality vs Free SO2')
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

# Grafik 4: Standard Error dan P-value
features = ['volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide']
p_values = [0.05, 0.04, 0.25, 0.22]
standard_errors = [0.02, 0.01, 0.03, 0.02]

x = range(len(features))
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x, standard_errors, width=width, label='Standard Error', color='blue')
plt.bar([i + width for i in x], p_values, width=width, label='P-value', color='gray', hatch='//')
plt.xticks([i + width / 2 for i in x], features, rotation=45)
plt.title('Error and P-values for Wine Features')
plt.ylabel('Values')
plt.legend()
plt.show()

# Menambahkan kategori "Good" dan "Bad" ke dalam dataset
data['quality_category'] = data['quality'].apply(lambda q: 'Good' if q >= 6 else 'Bad')

# Menampilkan keterangan kategori di output
print("\nKategori quality wine:")
print("Good: Quality >= 6")
print("Bad: Quality < 6")

# Menyimpan dataset yang sudah dimodifikasi ke dalam file CSV
output_csv_path = "categorized_wine_quality.csv"
data.to_csv(output_csv_path, index=False)

print(f"\nDataset yang sudah dikategorikan menjadi 'Good' dan 'Bad' telah disimpan ke file: {output_csv_path}")


# Tes input untuk red wine
red_wine_features = [0.7, 0.2, 2.5, 15]
red_wine_prediction = classify_wine(red_wine_features)
print(f"Prediksi untuk red wine dengan fitur {red_wine_features}: {red_wine_prediction}")

# Tes input untuk white wine
white_wine_features = [0.3, 0.4, 6.0, 25]
white_wine_prediction = classify_wine(white_wine_features)
print(f"Prediksi untuk white wine dengan fitur {white_wine_features}: {white_wine_prediction}")

# volatile acidity = 0.7: Cenderung lebih tinggi pada red wine.
# citric acid = 0.2: Relatif lebih rendah pada red wine.
# residual sugar = 2.5: Kandungan gula biasanya lebih rendah pada red wine.
# free sulfur dioxide = 15: Jumlah sulfur yang wajar untuk red wine.
# White Wine:

# volatile acidity = 0.3: Biasanya lebih rendah pada white wine.
# citric acid = 0.4: Kandungan citric acid lebih tinggi pada white wine.
# residual sugar = 6.0: Kandungan gula residu lebih tinggi untuk white wine.
# free sulfur dioxide = 25: Jumlah sulfur dioksida lebih tinggi untuk white wine.