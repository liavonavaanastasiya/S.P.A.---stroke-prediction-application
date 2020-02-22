######## FINAL PROJECT########

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import joblib

MODEL_FILEPATH = "model.pkl"

app = Flask(__name__) #Initialize the flask App

# load saved model
model = joblib.load(MODEL_FILEPATH)

# model = pickle.load(open(MODEL_FILEPATH, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.to_dict()
    
    features = {k:(float(v) if v.isdigit() else v) for (k,v) in features.items()}
    
    features['bmi'] = float(features["weight"]) / (features['height']/100)**2
    features.pop("height")
    features.pop("weight")
    
    features = {k:[v] for (k,v) in features.items()}
    
    features = pd.DataFrame.from_dict(features)
    
    features = pd.DataFrame(features, columns = ['gender', 'age', 'bmi', 'avg_glucose_level', 'hypertension',
       'heart_disease', 'ever_married', 'work_type', 'smoking'])
    
#     features.to_csv('new_data.csv', mode='w')
#     features = pd.read_csv('new_data.csv')
# #     features = features.iloc[-1,:]
    
    prediction = model.predict(features)
    
    probability = model.predict_proba(features)[:,1]
    
    
    
    if prediction == 0:
        
        return render_template('index.html', prediction_text=f'Patient is OK')
    
        
    else: out = f'Patient is in danger!!! \n  Probability of stroke is {probability}'

    return render_template('index.html', prediction_text=out)
if __name__ == "__main__":
    app.run()
