import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
# Dosyanın yolu
file_path = "C:/Users/TR/Desktop/Kitap.csv"  # Yüklediğiniz dosyanın yolunu buraya koyun

# CSV dosyasını okuma
data = pd.read_csv(file_path, sep=';')

# Virgülleri noktaya çevirme ve sayısal verilere dönüştürme
numerical_columns = ['Alkol (%)', 'Sabit Asitlik (g/L)', 'Kalan Şeker (g/L)', 'Toplam Kükürt Dioksit (mg/L)']
for column in numerical_columns:
    # Eğer veri tipinin string olduğunu kontrol ederek virgül yerine nokta koyma
    if data[column].dtype == 'object':
        data[column] = data[column].str.replace(',', '.').astype(float)
    else:
        data[column] = data[column].astype(float)

# Kategorik sütunların sayısal hale getirilmesi için LabelEncoder kullanma
label_encoder = LabelEncoder()

# Kategorik sütunları encode etme
data['Tür'] = label_encoder.fit_transform(data['Tür'])
data['Üzüm Türü'] = label_encoder.fit_transform(data['Üzüm Türü'])
data['İklim'] = label_encoder.fit_transform(data['İklim'])

# Sayısal verileri MinMaxScaler ile normalize etme
scaler = MinMaxScaler()

# Sayısal sütunları seçme (normalizasyon yapılacak)
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])