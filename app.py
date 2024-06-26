from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        ssc_p = float(request.form['ssc_p'])
        hsc_p = float(request.form['hsc_p'])
        hsc_b = request.form['hsc_b']
        hsc_b_other = request.form.get('hsc_b_other', '')  # Retrieve board name if "Others" selected
        hsc_s = request.form['hsc_s']
        degree_p = float(request.form['degree_p'])
        degree_t = request.form['degree_t']
        branch = request.form.get('branch', '')  # Optional field based on degree type
        specialisation = request.form.get('specialisation', '')  # Optional field based on branch
        workex = int(request.form['workex'])

        # Process board of high school
        if hsc_b == "Others":
            hsc_b_Central = 0
            hsc_b_Others = 1
        else:
            hsc_b_Central = 1
            hsc_b_Others = 0

        # Process stream of high school
        if hsc_s == "Others":
            hsc_s_MPC = 0
            hsc_s_BiPC = 0
            hsc_s_MEC = 0
            hsc_s_CEC = 0
            hsc_s_Others = 1
        else:
            hsc_s_MPC = 1 if hsc_s == "MPC" else 0
            hsc_s_BiPC = 1 if hsc_s == "BiPC" else 0
            hsc_s_MEC = 1 if hsc_s == "MEC" else 0
            hsc_s_CEC = 1 if hsc_s == "CEC" else 0
            hsc_s_Others = 0

        # Process type of undergraduate degree
        if degree_t == "Others":
            degree_t_BT = 0
            degree_t_BBA = 0
            degree_t_BA = 0
            degree_t_BSc = 0
            degree_t_Others = 1
        else:
            degree_t_BT = 1 if degree_t == "BTech" else 0
            degree_t_BBA = 1 if degree_t == "BBA" else 0
            degree_t_BA = 1 if degree_t == "BA" else 0
            degree_t_BSc = 1 if degree_t == "BSc" else 0
            degree_t_Others = 0

        # Process branch (optional)
        if degree_t == "BTech":
            if branch == "CSE":
                branch_CSE = 1
                branch_ECE = 0
                branch_EEE = 0
                branch_BioTech = 0
                branch_Mechanical = 0
                branch_Civil = 0
            elif branch == "ECE":
                branch_CSE = 0
                branch_ECE = 1
                branch_EEE = 0
                branch_BioTech = 0
                branch_Mechanical = 0
                branch_Civil = 0
            elif branch == "EEE":
                branch_CSE = 0
                branch_ECE = 0
                branch_EEE = 1
                branch_BioTech = 0
                branch_Mechanical = 0
                branch_Civil = 0
            elif branch == "BioTech":
                branch_CSE = 0
                branch_ECE = 0
                branch_EEE = 0
                branch_BioTech = 1
                branch_Mechanical = 0
                branch_Civil = 0
            elif branch == "Mechanical":
                branch_CSE = 0
                branch_ECE = 0
                branch_EEE = 0
                branch_BioTech = 0
                branch_Mechanical = 1
                branch_Civil = 0
            elif branch == "Civil":
                branch_CSE = 0
                branch_ECE = 0
                branch_EEE = 0
                branch_BioTech = 0
                branch_Mechanical = 0
                branch_Civil = 1
            else:
                branch_CSE = 0
                branch_ECE = 0
                branch_EEE = 0
                branch_BioTech = 0
                branch_Mechanical = 0
                branch_Civil = 0
        else:
            branch_CSE = 0
            branch_ECE = 0
            branch_EEE = 0
            branch_BioTech = 0
            branch_Mechanical = 0
            branch_Civil = 0

        # Process specialisation (optional)
        if branch == "CSE":
            if specialisation == "Artificial Intelligence":
                specialisation_AI = 1
                specialisation_ML = 0
                specialisation_DS = 0
                specialisation_Robotics = 0
            elif specialisation == "Machine Learning":
                specialisation_AI = 0
                specialisation_ML = 1
                specialisation_DS = 0
                specialisation_Robotics = 0
            elif specialisation == "Data Science":
                specialisation_AI = 0
                specialisation_ML = 0
                specialisation_DS = 1
                specialisation_Robotics = 0
            elif specialisation == "Robotics":
                specialisation_AI = 0
                specialisation_ML = 0
                specialisation_DS = 0
                specialisation_Robotics = 1
            else:
                specialisation_AI = 0
                specialisation_ML = 0
                specialisation_DS = 0
                specialisation_Robotics = 0
        else:
            specialisation_AI = 0
            specialisation_ML = 0
            specialisation_DS = 0
            specialisation_Robotics = 0

        # Make prediction
        scaled = scaler.transform(np.array([
            gender, ssc_p, hsc_p, degree_p, workex,
            hsc_b_Central, hsc_b_Others, hsc_s_MPC, hsc_s_BiPC, hsc_s_MEC, hsc_s_CEC, hsc_s_Others,
            degree_t_BT, degree_t_BBA, degree_t_BA, degree_t_BSc, degree_t_Others,
            branch_CSE, branch_ECE, branch_EEE, branch_BioTech, branch_Mechanical, branch_Civil,
            specialisation_AI, specialisation_ML, specialisation_DS, specialisation_Robotics
        ]).reshape(1, -1))

        prediction = round(model.predict(scaled)[0], 2)
        if prediction < 0:
            prediction = 0

        return render_template('prediction.html', result=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)

