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

    input_data = [sepal_length, sepal_width, petal_length, petal_width]

    # format data
    format_data = [float(i) for i in input_data]

    # filter data
    filter_data = np.array(format_data).reshape(1, -1)

    prediction = svm_model(filter_data)

    # render template
    return render_template('index.html', sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                           petal_width=petal_width, input_data=format_data, prediction=prediction)


def svm_model(value):
    # read datasets and get X, y
    data = pd.read_csv('data/iris.csv', delimiter=',')
    X = data.iloc[:, 0:4]
    y = data.variety

    # building model
    model = svm.SVC(kernel='rbf')
    model.fit(X, y)

    # result
    prediction = model.predict(value)
    return prediction


if __name__ == '__main__':
    app.run(debug=True)
