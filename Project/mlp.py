import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import tanimlama
data = tanimlama.data
data_X = data.loc[:, data.columns != "Tür"]
data_Y = data[["Tür"]]

print("data_X info:\n")
data_X.info()
print("\ndata_Y info:\n")
data_Y.info()
data_Y["Tür"].value_counts()

train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y,
                                                    test_size=0.5,
                                                    stratify=data_Y,
                                                    random_state=0)

train_X.reset_index(drop=True, inplace=True);
test_X.reset_index(drop=True, inplace=True);
train_Y.reset_index(drop=True, inplace=True);
test_Y.reset_index(drop=True, inplace=True);

feature_names = train_X.columns

scaler = StandardScaler()

# fit to train_X
scaler.fit(train_X)

# transform train_X
train_X = scaler.transform(train_X)
train_X = pd.DataFrame(train_X, columns = feature_names)

# transform test_X
test_X = scaler.transform(test_X)
test_X = pd.DataFrame(test_X, columns = feature_names)

clf1 = MLPClassifier(solver="adam", max_iter=5000, activation = "relu",
                    #hidden_layer_sizes = (12), 
                    hidden_layer_sizes = (32),
                    alpha = 0.01,
                    batch_size = 25,
                    learning_rate_init = 0.001,
                    random_state=2)

clf1.fit(train_X, train_Y.values.ravel())
clf2 = MLPClassifier(solver="adam", max_iter=5000, activation = "relu",
                    #hidden_layer_sizes = (12), 
                    hidden_layer_sizes = (32, 32),
                    alpha = 0.01,
                    batch_size = 25,
                    learning_rate_init = 0.001,
                    random_state=2)

clf2.fit(train_X, train_Y.values.ravel())
clf3 = MLPClassifier(solver="adam", max_iter=5000, activation = "relu",
                    #hidden_layer_sizes = (12), 
                    hidden_layer_sizes = (32,32,32),
                    alpha = 0.01,
                    batch_size = 25,
                    learning_rate_init = 0.001,
                    random_state=2)

clf3.fit(train_X, train_Y.values.ravel())
classification_report1=classification_report(test_Y, clf1.predict(test_X),
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report1)
print("-----------------------------------------------")
classification_report2=classification_report(test_Y, clf2.predict(test_X),
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report2)
print("------------------------------------------------------")
classification_report3=classification_report(test_Y, clf3.predict(test_X),
                            digits = 4,
                            target_names=["Kırmızı",
                                          "Beyaz"])
print(classification_report3)