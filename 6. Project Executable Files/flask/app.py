from flask import Flask, render_template, request 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))
# ct = joblib.load('feature_values.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pred')
def predict():
    return render_template("index.html")

@app.route('/out', methods =["POST"])
def output():
    age = request.form.get("age")
    gender = request.form.get("gender",)
    self_employed = request.form.get("self_employed")
    family_history= request.form.get("family_history")
    work_interfere = request.form.get("work_interfere")
    no_employees = request.form.get("no_employees")
    remote_work = request.form.get("remote_work")
    tech_company =request.form.get("tech_company")
    benefits = request.form.get("benefits")
    care_options = request.form.get("care_options")
    wellness_program = request.form.get("wellness_program")
    seek_help = request.form.get("seek_help")
    anonymity = request.form.get("anonymity")
    leave = request.form.get("leave")
    mental_health_consequence = request.form.get("mental_health_consequence")
    phys_health_consequence = request.form.get("phys_health_consequence")
    coworkers = request.form.get("coworkers")
    supervisor = request.form.get("supervisor")
    mental_health_interview = request.form.get("mental_health_interview")
    phys_health_interview = request.form.get("phys_health_interview")
    mental_vs_physical = request.form.get("mental_vs_physical")
    obs_consequence = request.form.get("obs_consequence")
    
    df = pd.DataFrame({'age': [age], 'gender': [gender], 'self_employed':[self_employed], 'family_history':[family_history], 'work_interfere':[work_interfere], 'no_employees':[no_employees], 'remote_work':[remote_work], 'tech_company':[tech_company], 'benefits':[benefits], 'care_options':[care_options], 'wellness_program':[wellness_program], 'seek_help':[seek_help], 'anonymity':[anonymity], 'leave':[leave], 'mental_health_consequence':[mental_health_consequence], 'phys_health_consequence':[phys_health_consequence], 'coworkers':[coworkers], 'supervisor':[supervisor], 'mental_health_interview':[mental_health_interview], 'phys_health_interview':[phys_health_interview], 'mental_vs_physical':[mental_vs_physical], 'obs_consequence':[obs_consequence]})
    
    prediction = model.predict(df)  # predict using the DataFrame
    
    output = "the prediction is " + str(['yes', 'no'][np.argmax(pred)])
    
    return render_template(output, "output.html")

if __name__ == '_main_':
    app.run(debug = False)