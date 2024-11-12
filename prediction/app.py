from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained models and scaler
with open('rf_placement_model.pkl', 'rb') as f:
    rf_placement_model = pickle.load(f)

with open('rf_salary_model.pkl', 'rb') as f:
    rf_salary_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # An HTML form to take user inputs

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the user input
    cgpa = float(request.form['cgpa'])
    major_projects = int(request.form['major_projects'])
    workshops = int(request.form['workshops'])
    mini_projects = int(request.form['mini_projects'])
    skills = int(request.form['skills'])
    comm_skill = float(request.form['comm_skill'])
    internship = int(request.form['internship'])  # 1 for Yes, 0 for No
    hackathon = int(request.form['hackathon'])  # 1 for Yes, 0 for No
    tenth_percentage = int(request.form['tenth_percentage'])
    twelfth_percentage = int(request.form['twelfth_percentage'])
    backlogs = int(request.form['backlogs'])

    # Create feature array
    features = np.array([[cgpa, major_projects, workshops, mini_projects, skills, comm_skill, internship, hackathon,
                          tenth_percentage, twelfth_percentage, backlogs]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict placement status
    placement_pred = rf_placement_model.predict(features_scaled)
    placement_status = 'Placed' if placement_pred[0] == 1 else 'Not Placed'

    # Predict salary if placed
    if placement_pred[0] == 1:
        salary_pred = rf_salary_model.predict(features_scaled)
        salary = round(salary_pred[0], 2)
    else:
        salary = 0  # If not placed, salary is 0

    return render_template('result.html', placement_status=placement_status, salary=salary)

if __name__ == '__main__':
    app.run(debug=True)
