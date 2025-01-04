import tanimlama  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# s.py dosyasındaki data verisini al
data = tanimlama.data

# Veriyi ayırma
X, y = data.iloc[:, 1:], data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

# Veriyi standartlaştırma
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)  

# Naive Bayes sınıflandırıcısı
classifierNB = GaussianNB()

# Model eğitimi
classifierNB.fit(X_train_std, y_train)

# Tahmin
y_predNB = classifierNB.predict(X_test_std)

# Sonuçları yazdırma
print("------------------------------------------------------")
print("Naive Bayes Sonuçları:")
classification_reportNB = classification_report(y_test, y_predNB, digits=4, target_names=["Kırmızı", "Beyaz"])

# Naive Bayes sonuçlarını yazdırma
print(classification_reportNB)
