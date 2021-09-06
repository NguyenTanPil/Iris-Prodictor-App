import pandas as pd
from sklearn import svm
import pickle


def svm_model():
    data = pd.read_csv('iris.csv', delimiter=',')
    X = data.iloc[:, 0:4]
    y = data.variety

    # building model
    model = svm.SVC(kernel='rbf')
    model.fit(X, y)

    # save model
    with open('svm_model.pkl', 'wb') as file:
        pickle.dump(model, file)


svm_model()
