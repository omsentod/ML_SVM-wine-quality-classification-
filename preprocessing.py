import pandas as pd

# membaca dataset
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

print("=========== Dataset Awal ===========")
# menampilkan dataset awal
print("Red Wine : ", red_wine.head())
print("White Wine : ", white_wine.head())

print("\nRed Wine awal : \n", red_wine.shape)
print("White Wine awal : \n", white_wine.shape)

print("\n=========== Pengecekan Missing value ===========")
# melakuakn pengecekan missing value dan penanganan missing value
# 1. Cek missing value
print("Red Wine Missing Value : \n", red_wine.isnull().sum())
print("White Wine Missing Value : \n", white_wine.isnull().sum())

# 2. Handling missing value dengan melakuakn drop pada missing value yang terdeteksi
red_wine = red_wine.dropna()
white_wine = white_wine.dropna()
# mengecek kembali missing value
print("\nRed Wine Missing Value setelah penanganan : \n", red_wine.isnull().sum())
print("White Wine Missing Value setelah penanganan : \n", white_wine.isnull().sum())

print("\nRed Wine setelah dihilangkan missing value : ", red_wine.shape)
print("White Wine setelah dihilangkan missing value : ", white_wine.shape)

print("\n=========== Pengecekan Outlier ===========")
# melakukan pengecekan dan penanganan outlier menggunakan quartile
# 1. Cek outlier red wine
print("Red Wine Describe : ", red_wine.describe())

Q1 = red_wine.quantile(0.25)
Q3 = red_wine.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 2. Handling outlier red wine dengan melakukan drop pada data yang terdeteksi sebagai outlier
outlier = ((red_wine < lower_bound) | (red_wine > upper_bound)).any(axis=1)
red_wine = red_wine[~outlier]
outliers_summary = outlier.sum()
print("\nJumlah Outlier red wine :", outliers_summary)
print("dataset setelah dihilangkan outlier : ", red_wine.shape)

# 3. Cek outlier white wine
print("\nWhite Wine Describe : ", white_wine.describe())

Q1 = white_wine.quantile(0.25)
Q3 = white_wine.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 4. Handling outlier white wine dengan melakukan drop pada data yang terdeteksi sebagai outlier
outlier = ((white_wine < lower_bound) | (white_wine > upper_bound)).any(axis=1) #mendeteksi outlier
white_wine = white_wine[~outlier] #menghilangkan outlier
outliers_summary = outlier.sum()
print("\nJumlah Outlier white wine :", outliers_summary)
print("dataset setelah dihilangkan outlier :", white_wine.shape)

print("\n=========== Dataset Akhir ===========")
# menampilkan dataset akhir
print("dataset akhir red wine : ", red_wine.head())
print("dataset akhir white wine : ", white_wine.head())

# menyimpan dataset akhir
red_wine.to_csv('winequality-red_clean_preprocessing.csv', index=False)
white_wine.to_csv('winequality-white_clean_preprocessing.csv', index=False)
print("\ndataset di simpan pada file winequality-red_clean_preprocessing.csv dan winequality-white_clean_preprocessing.csv")