import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#Lendo os dados

sismica = pd.read_csv("~/sismica.csv", nrows=50)

#Separando os dados das classes

data = sismica.iloc[:,:-1].values
target = sismica.iloc[:,-1].values

#Gerando a divisão

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=990)

#Loop para percorrer os folds


for train, test in rkf.split(data, target):
    X_train = data[train]
    X_test = data[test]
    y_train = target[train]
    y_test = target[test]


model = KNeighborsClassifier(n_neighbors=2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)

acc = str(accuracy_score(y_test, y_pred))

print(cm)

print(acc)
