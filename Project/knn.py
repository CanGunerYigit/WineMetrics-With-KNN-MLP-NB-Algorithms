import tanimlama  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


data = tanimlama.data 

X,y = data.iloc[:,1:], data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state= 0, stratify = y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

classifier1 = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean')
classifier2 = KNeighborsClassifier(n_neighbors = 7,  metric='euclidean')
classifier3 = KNeighborsClassifier(n_neighbors = 11,  metric='euclidean')

classifier1.fit(X_train_std, y_train)
classifier2.fit(X_train_std, y_train)
classifier3.fit(X_train_std, y_train)

y_pred1 = classifier1.predict(X_test_std)
y_pred2 = classifier2.predict(X_test_std)
y_pred3 = classifier3.predict(X_test_std)


cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
print(y_test)
print("----------------------")
print(y_pred1,y_pred2,y_pred3)
print("----------------------")
print(cm1,cm2,cm3)
print("----------------------")
accuracy_score1=accuracy_score(y_test, y_pred1)
print(accuracy_score1)
accuracy_score2=accuracy_score(y_test, y_pred2)
print(accuracy_score2)
accuracy_score3=accuracy_score(y_test, y_pred3)
print(accuracy_score3)

print("------------------------------------------------------")
classification_report1=classification_report(y_test, y_pred1,zero_division=0,
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report1)
classification_report2=classification_report(y_test, y_pred2,zero_division=0,
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report2)
classification_report3=classification_report(y_test, y_pred3,zero_division=0,
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report3)
