######## FINAL PROJECT########

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, Markup
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

@app.route('/about')
def about():
    return render_template('about.html')

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
    probability_proc = probability[0]*100
    
    recomendations = []

    # 1st in feature importance list 
    if features['bmi'][0] > 25:
        recomendations.append('Please, note that your BMI level is above normal. Even small amounts of weight loss can bring a range of health benefits such as stroke risk reduction, improved blood pressure, cholesterol and blood sugar levels. Making changes to your diet and physical activity can also have health benefits that are independent of weight loss, including increased energy, better sleep and reduced risk of depression.')

    # 3rd in feature importance list  
    if features['avg_glucose_level'][0] > 100:
        recomendations.append('The American Diabetes Association recommend target glucose level to be below 99 mg/dL before eating and below 130 mg/dL for a person with diabetes. Instead of targeting a specific level, the aim of managing blood sugar is to keep the levels within a healthy range.')

    # 4th in feature importance list  
    if features['heart_disease'][0] == 1:
        recomendations.append('It is important to understand that the link between heart disease and stroke is significant, several types of heart disease are risk factors for stroke. Please, visit your doctor to discuss lifestyle choices that will reduce your risk of stroke.')

    # 5th in feature importance list 
    if features['hypertension'][0] == 1:
        recomendations.append('Take care, high blood pressure damages arteries throughout the body, creating conditions where they can burst or clog more easily. Weakened arteries in the brain, resulting from high blood pressure, put you at a much higher risk for stroke — which is why managing high blood pressure is critical to reduce your chance of having a stroke. Healthy lifestyle changes may help you to keep it down.')   

    # 6th in feature importance list 
    if features['smoking'][0] == 3:
        recomendations.append('Dear, customer, we want you to know that tobacco smoking is a major risk factor for stroke. Current smokers have a 2-4 times increased risk of stroke compared with nonsmokers or those who have quit for >10 years')

    recomendations = "\n".join(recomendations)
    # >>>>>>> fc463c26f733dda659470fbefeed6a013d362932
    
    if prediction == 0:
        # for healthy people   
        if (features['bmi'][0] <= 25)  and (features['avg_glucose_level'][0] <= 100) and (features['heart_disease'][0] != 1) and (features['hypertension'][0] != 1) and (features['smoking'][0] != 3):
            recomendations = 'Congratulations, stroke is unlikely. Please, take care and remember, that You can reduce the risk of stroke by avoiding smoking and stress, eating helthy food, exercising and doing regular check-ups'
        return render_template('result.html', prediction_text=f"Probability of stroke is less than {probability_proc:.1f}%", recomendations=recomendations)
    
    else: 
        out = f'You are at risk! Probability of stroke is {probability_proc:.1f}% \n To reduce the risk, please follow the next recommendations:'

    return render_template('result.html', prediction_text=out, recomendations=recomendations)

if __name__ == "__main__":
    app.run()
