import pandas as pd
import numpy as np
import flask
import pickle
from flask import Flask, render_template, redirect, url_for, request, session, flash

app = Flask(__name__)


@app.route('/home')
def home():
    return flask.render_template('home.html')


@app.route('/about')
def about():
    return flask.render_template('about.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)


@app.route('/data-vis')
def data_vis():
    return flask.render_template('data-vis.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.')
    return redirect(url_for('login'))


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 6)
    loaded_model = pickle.load(open("random_forest_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predict', methods=['POST'])
def result():
 if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    result = ValuePredictor(to_predict_list)
    prediction = str(result)
    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

