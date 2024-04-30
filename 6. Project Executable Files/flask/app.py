from flask import Flask, render_template, request 
import pickle, joblib 
import pandas as pd

app = Flask(__name__)


model = pickle.load(open("model.pkl",'rb'))
ct = joblib.load('feature_values.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pred')
def predict():
    return render_template("index.html")

@app.route('/out', methods =["POST"])
def output():
    # age = request.form.get("age")
    # gender = request.form.get("gender",)
    # self_employed = request.form.get("self_employed")
    # family_history= request.form.get("family_history")
    # work_interfere = request.form.get("work_interfere")
    # no_employees = request.form.get("no_employees")
    # remote_work = request.form.get("remote_work")
    # tech_company =request.form.get("tech_company")
    # benefits = request.form.get("benefits")
    # care_options = request.form.get("care_options")
    # wellness_program = request.form.get("wellness_program")
    # seek_help = request.form.get("seek_help")
    # anonymity = request.form.get("anonymity")
    # leave = request.form.get("leave")
    # mental_health_consequence = request.form.get("mental_health_consequence")
    # phys_health_consequence = request.form.get("phys_health_consequence")
    # coworkers = request.form.get("coworkers")
    # supervisor = request.form.get("supervisor")
    # mental_health_interview = request.form.get("mental_health_interview")
    # phys_health_interview = request.form.get("phys_health_interview")
    # mental_vs_physical = request.form.get("mental_vs_physical")
    # obs_consequence = request.form.get("obs_consequence")

    # data = [[age, gender, self_employed, family_history, work_interfere, no_employees, remote_work, tech_company, benefits, care_options, wellness_program, seek_help, anonymity, leave, mental_health_consequence, phys_health_consequence, coworkers, supervisor, mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence]]

    # feature_cols = ['Age', 'Gender', 'self_employed', 'family_history',

    # 'work_interfere', 'no_employees', 'remote_work', 'tech_company',

    # 'benefits', 'care_options', 'wellness_program', 'seek_help',

    # 'anonymity', 'leave', 'mental_health_consequence',

    # 'phys_health_consequence', 'coworkers', 'supervisor',

    # 'mental_health_interview', 'phys_health_interview',

    # 'mental_vs_physical', 'obs_consequence']

    # pred =model.predict(ct.transform(pd.DataFrame (data, columns=feature_cols)))

    # pred = pred[0]

    # if pred:
        return render_template("output.html",y="This person requires mental health treatment ")
    # else:
        # return render_template("output.html",y="This person doesn't require mental health treatment")


if __name__ == '_main_':
    app.run(debug = True)
