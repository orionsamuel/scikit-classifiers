
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.naive_bayes import GaussianNB as naive
from sklearn.neighbors import KNeighborsClassifier as knn
from mlxtend.classifier import StackingClassifier

clf = []

#Lendo os dados

sismica = pd.read_csv("~/sismica.csv", nrows=50)

#Separando os dados das classes

data = sismica.iloc[:,:-1].values
target = sismica.iloc[:,-1].values

#Gerando a divisão

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1967)

#Loop para percorrer os folds


for train, test in rkf.split(data, target):
    X_train = data[train]
    X_test = data[test]
    y_train = target[train]
    y_test = target[test]

for i in range(10):
    if(i < 3):
        clf.append(knn(n_neighbors=2))
    elif(i >=3 and i < 8):
        clf.append(naive())
    else:
        clf.append(mlp(solver='sgd', momentum=0.8, hidden_layer_sizes=(150), learning_rate='constant', learning_rate_init=0.1, max_iter=500, random_state=870))

meta = naive()

sclf = StackingClassifier(classifiers=[clf[0], clf[0], clf[1], clf[2], clf[3], clf[4], clf[5], clf[6], clf[7], clf[8], clf[9]], meta_classifier=meta)

sclf.fit(X_train, y_train)

y_pred = sclf.predict(X_test)

print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)

acc = str(accuracy_score(y_test, y_pred))

print(cm)

print(acc)
