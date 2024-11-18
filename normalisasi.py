import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Membaca dataset yang telah dipreprocessing
red_wine = pd.read_csv('winequality-red_clean_preprocessing.csv')
white_wine = pd.read_csv('winequality-white_clean_preprocessing.csv')

# Inisialisasi MinMax Scaler dengan rentang [0, 1]
scaler = MinMaxScaler()

# Normalisasi dataset red wine kecuali kolom 'quality'
red_wine_features = red_wine.drop(columns=['quality'])  # Memisahkan fitur dari kolom quality
red_wine_normalized = pd.DataFrame(scaler.fit_transform(red_wine_features), columns=red_wine_features.columns)
red_wine_normalized['quality'] = red_wine['quality']  # Menambahkan kembali kolom quality
print("=========== Red Wine Normalized ===========")
print(red_wine_normalized.head())

# Normalisasi dataset white wine kecuali kolom 'quality'
white_wine_features = white_wine.drop(columns=['quality'])  # Memisahkan fitur dari kolom quality
white_wine_normalized = pd.DataFrame(scaler.fit_transform(white_wine_features), columns=white_wine_features.columns)
white_wine_normalized['quality'] = white_wine['quality']  # Menambahkan kembali kolom quality
print("\n=========== White Wine Normalized ===========")
print(white_wine_normalized.head())

# Menyimpan hasil normalisasi ke dalam file CSV baru
red_wine_normalized.to_csv('winequality-red_normalized.csv', index=False)
white_wine_normalized.to_csv('winequality-white_normalized.csv', index=False)
print("\nDataset yang telah dinormalisasi disimpan sebagai 'winequality-red_normalized.csv' dan 'winequality-white_normalized.csv'")
