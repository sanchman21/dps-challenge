from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
# Initalise the Flask app
app = Flask(__name__)

model0 = pickle.load(open('xgb_reg0.pkl', "rb"))
model1 = pickle.load(open('xgb_reg1.pkl', "rb"))
model2 = pickle.load(open('xgb_reg2.pkl', "rb"))

def encode_month(data):
    data['Month' + '_sin'] = np.sin(2 * np.pi * int(data['Month'])/12)
    data['Month' + '_cos'] = np.cos(2 * np.pi * int(data['Month'])/12)

def encode_year(data):
    data['Year'] = int(data['Year']) - 2000

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    if 'Type' in data_unseen.columns:
        data_unseen.drop('Type', axis=1, inplace=True)
    if 'Category' in data_unseen.columns:
        category = data_unseen['Category'].iloc[0]
        data_unseen.drop('Category', axis=1, inplace=True)
    else:
        category = 'Alkoholunfälle'
    encode_month(data_unseen)
    encode_year(data_unseen)
    data_unseen.drop('Month', axis=1, inplace=True)
    if category == 'Fluchtunfälle':
        prediction = model1.predict(data_unseen)
    elif category == 'Verkehrsunfälle':
        prediction = model2.predict(data_unseen)
    else:
        prediction = model0.predict(data_unseen)
    output = {'prediction': int(prediction[0])}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)