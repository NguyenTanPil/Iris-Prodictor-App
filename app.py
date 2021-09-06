import joblib
import pandas as pd
import numpy as np
from sklearn import svm

from flask import Flask, render_template, url_for, request
from flask_material import Material

app = Flask(__name__)
Material(app)


@app.route('/')
def index():
    return render_template('index.html', sepal_length=0.0, sepal_width=0.0, petal_length=0.0,
                           petal_width=0.0)


@app.route('/preview')
def preview():
    df = pd.read_csv('data/iris.csv')
    return render_template('preview.html', df_review=df)


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
        model_choose = request.form['model_choose']

    input_data = [sepal_length, sepal_width, petal_length, petal_width]

    # format data
    format_data = [float(i) for i in input_data]

    # filter data
    filter_data = np.array(format_data).reshape(1, -1)

    # Loading model
    model_name = ''
    if model_choose == 'knn':
        model = joblib.load('data/knn_model.pkl')
        model_name = 'KNN'
    elif model_choose == 'naive_bayes':
        model = joblib.load('data/naive_bayes_model.pkl')
        model_name = 'Naive Bayes'
    elif model_choose == 'decision_tree':
        model = joblib.load('data/decision_tree_model.pkl')
        model_name = 'Decision Tree'
    else:
        model = joblib.load('data/svm_model.pkl')
        model_name = 'SVM'

    prediction = model.predict(filter_data)


    # render template
    return render_template('index.html', sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                           petal_width=petal_width, input_data=format_data, model_name=model_name, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
