import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle


def svm_model():
    data = pd.read_csv('iris.csv', delimiter=',')
    X = data.iloc[:, 0:4]
    y = data.variety

    # building model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    # save model
    with open('knn_model.pkl', 'wb') as file:
        pickle.dump(model, file)


svm_model()
