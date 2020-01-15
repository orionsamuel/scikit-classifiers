import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB as naive

#Lendo os dados

sismica = pd.read_csv("~/sismica.csv")

#Separando os dados das classes

data = sismica.iloc[:,:-1].values
target = sismica.iloc[:,-1].values

#Gerando a divisão

kfold = StratifiedKFold(n_splits=10)

#Loop para percorrer os folds

for train, test in kfold.split(data, target):
    X_train = data[train]
    X_test = data[test]
    y_train = target[train]
    y_test = target[test]

model = naive()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

cm = confusion_matrix(y_test, y_pred)

acc = str(accuracy_score(y_test, y_pred))

print(cm)

print(acc)


