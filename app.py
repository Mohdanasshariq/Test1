from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']
    val5 = request.form['sqft_living']
    val6 = request.form['sqft_above']
    val7 = request.form['grade']
    arr = np.array([[val1, val2, val3, val4, val5, val6, val7]])
    arr = scale.transform(arr)
    pred = model.predict(arr)

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
