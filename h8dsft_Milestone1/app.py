import flask
from flask.globals import request
import numpy as np
import pandas as pd
import pickle

app = flask.Flask(__name__, template_folder = 'templates')

# render template
@app.route('/')
def main():
    return(flask.render_template('main.html'))

# load pickle file
model = pickle.load(open('model/best_model_svm.pkl', 'rb'))
scaler = pickle.load(open('model/scaling_pipeline.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering prediction result.
    '''
    # retreive data
    age = int(request.form.get('age'))
    education = request.form.get('education')
    duration = int(request.form.get('duration'))
    campaign = int(request.form.get('campaign'))
    previous = int(request.form.get('previous'))
    emp_var_rate = float(request.form.get('emp_var_rate'))
    job = request.form.get('job')
    marital = request.form.get('marital')
    default = request.form.get('default')
    housing = request.form.get('housing')
    loan = request.form.get('loan')
    contact = request.form.get('contact')
    poutcome = request.form.get('poutcome')
    month = int( request.form.get('month'))
    day_of_week = int(request.form.get('day_of_week'))


    # pre processing one hot encoding that previously used get_dummies
    known_job = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    known_marital = ['divorced', 'married', 'single', 'unknown']
    known_default = ['no', 'unknown', 'yes']
    known_housing = ['no', 'unknown', 'yes']
    known_loan = ['no', 'unknown', 'yes']
    known_contact = ['cellular', 'telephone']
    known_poutcome = ['failure', 'nonexistent', 'success']


    job_type = pd.Series([job])
    job_type = pd.Categorical(job_type, categories = known_job)
    job_input = pd.get_dummies(job_type, prefix = 'job', drop_first=True)

    marital_type = pd.Series([marital])
    marital_type = pd.Categorical(marital_type, categories = known_marital)
    marital_input = pd.get_dummies(marital_type, prefix = 'job', drop_first=True)

    default_type = pd.Series([default])
    default_type = pd.Categorical(default_type, categories = known_default)
    default_input = pd.get_dummies(default_type, prefix = 'job', drop_first=True)

    housing_type = pd.Series([housing])
    housing_type = pd.Categorical(housing_type, categories = known_housing)
    housing_input = pd.get_dummies(housing_type, prefix = 'job', drop_first=True)

    loan_type = pd.Series([loan])
    loan_type = pd.Categorical(loan_type, categories = known_loan)
    loan_input = pd.get_dummies(loan_type, prefix = 'job', drop_first=True)

    contact_type = pd.Series([contact])
    contact_type = pd.Categorical(contact_type, categories = known_contact)
    contact_input = pd.get_dummies(contact_type, prefix = 'job', drop_first=True)

    poutcome_type = pd.Series([poutcome])
    poutcome_type = pd.Categorical(poutcome_type, categories = known_poutcome)
    poutcome_input = pd.get_dummies(poutcome_type, prefix = 'job', drop_first=True)

    
    # pre processing cyclical feature
    month_sin = np.sin(month*(2.*np.pi/12))
    month_cos = np.cos(month*(2.*np.pi/12))

    day_of_week_sin = np.sin(day_of_week*(2.*np.pi/12))
    day_of_week_cos = np.cos(day_of_week*(2.*np.pi/12))

    # concat new data
    onehot_result = list(pd.concat([job_input, marital_input, default_input, housing_input, loan_input, contact_input, poutcome_input], axis = 1).iloc[0])
    new_data = [[age, education, duration, campaign, previous, emp_var_rate] + onehot_result + [month_sin, month_cos, day_of_week_sin, day_of_week_cos]]

    scaled_input = scaler.transform(new_data)
    prediction = model.predict(scaled_input)

    output = {0: 'not subscribe', 1: 'subscribe'}

    return flask.render_template('results.html', prediction_text = 'The Consumer will {} the term deposit.'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run(debug = True)
