import knn  # knn.py dosyasını import et
import mlp  # mlp.py dosyasını import et
import nb  # nb.py dosyasını import et
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# KNN Sonuçları (knn.py dosyasındaki classification_report çıktıları)
print("KNN (k=3) Sonuçları:")
print(knn.cm1)  # knn.py dosyasındaki cm1 (KNN k=3 için confusion matrix)
print(knn.accuracy_score1)  # knn.py dosyasındaki accuracy_score1
print(knn.classification_report1)  # knn.py dosyasındaki classification_report1

print("\nKNN (k=7) Sonuçları:")
print(knn.cm2)  # knn.py dosyasındaki cm2 (KNN k=7 için confusion matrix)
print(knn.accuracy_score2)  # knn.py dosyasındaki accuracy_score2
print(knn.classification_report2)  # knn.py dosyasındaki classification_report2

print("\nKNN (k=11) Sonuçları:")
print(knn.cm3)  # knn.py dosyasındaki cm3 (KNN k=11 için confusion matrix)
print(knn.accuracy_score3)  # knn.py dosyasındaki accuracy_score3
print(knn.classification_report3)  # knn.py dosyasındaki classification_report3


# MLP Sonuçları (mlp.py dosyasındaki classification_report çıktıları)
print("\nMLP (32) Sonuçları:")
print(mlp.classification_report1)

print("\nMLP (32, 32) Sonuçları:")
print(mlp.classification_report2)

print("\nMLP (32, 32, 32) Sonuçları:")
print(mlp.classification_report3)


# Naive Bayes Sonuçları (nb.py dosyasındaki classification_report çıktısı)
print("\nNaive Bayes Sonuçları:")
print(nb.classification_reportNB)
performance_data = {
    "Algorithm": ["KNN (k=3)", "KNN (k=7)", "KNN (k=11)", "MLP (32)", "MLP (32,32)", "MLP (32,32,32)", "Naive Bayes"],
    "Accuracy": [
        accuracy_score(knn.y_test, knn.y_pred1),
        accuracy_score(knn.y_test, knn.y_pred2),
        accuracy_score(knn.y_test, knn.y_pred3),   
        accuracy_score(mlp.test_Y, mlp.clf1.predict(mlp.test_X)),
        accuracy_score(mlp.test_Y, mlp.clf2.predict(mlp.test_X)),
        accuracy_score(mlp.test_Y, mlp.clf3.predict(mlp.test_X)),
        accuracy_score(nb.y_test, nb.y_predNB)
    ],
    "Precision": [
        classification_report(knn.y_test, knn.y_pred1, zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(knn.y_test, knn.y_pred2, zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(knn.y_test, knn.y_pred3, zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(mlp.test_Y, mlp.clf1.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(mlp.test_Y, mlp.clf2.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(mlp.test_Y, mlp.clf3.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["precision"],
        classification_report(nb.y_test, nb.y_predNB, zero_division=0, output_dict=True)["weighted avg"]["precision"]
    ],
    "Recall": [
        classification_report(knn.y_test, knn.y_pred1, zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(knn.y_test, knn.y_pred2, zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(knn.y_test, knn.y_pred3, zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(mlp.test_Y, mlp.clf1.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(mlp.test_Y, mlp.clf2.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(mlp.test_Y, mlp.clf3.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["recall"],
        classification_report(nb.y_test, nb.y_predNB, zero_division=0, output_dict=True)["weighted avg"]["recall"]
    ],
    "F1 Score": [
        classification_report(knn.y_test, knn.y_pred1, zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(knn.y_test, knn.y_pred2, zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(knn.y_test, knn.y_pred3, zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(mlp.test_Y, mlp.clf1.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(mlp.test_Y, mlp.clf2.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(mlp.test_Y, mlp.clf3.predict(mlp.test_X), zero_division=0, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(nb.y_test, nb.y_predNB, zero_division=0, output_dict=True)["weighted avg"]["f1-score"]
    ]
}

performance_df = pd.DataFrame(performance_data)

# Tablonun yazdırılması
print("\nPerformans Metrikleri Tablosu:")
print(performance_df.to_string(index=False))

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(performance_data["Algorithm"]))  # Algoritmalar için x ekseni pozisyonları
width = 0.2  # Çubuk genişliği

fig, ax = plt.subplots(figsize=(12, 6))

# Her bir metriği çiz
for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,  #  x pozisyonu
        performance_df[metric],  # Yükseklik değerleri
        width,  #  genişliği
        label=metric 
    )

# Grafiği özelleştir
ax.set_xlabel("Algorithms")
ax.set_ylabel("Scores")
ax.set_title("Şarap Metrik Tablosu")
ax.set_xticks(x + width * (len(metrics) - 1) / 2)  # X ekseni etiketlerinin merkezi hizalanması
ax.set_xticklabels(performance_data["Algorithm"], rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()

