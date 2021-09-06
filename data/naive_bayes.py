import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle


def svm_model():
    data = pd.read_csv('iris.csv', delimiter=',')
    X = data.iloc[:, 0:4]
    y = data.variety

    # building model
    model = GaussianNB()
    model.fit(X, y)

    # save model
    with open('naive_bayes_model.pkl', 'wb') as file:
        pickle.dump(model, file)


svm_model()
