 

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

new = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@new.route('/')
def home():
    return render_template('index.html')

@new.route('/predict',methods=['POST'])
def predict():
    '''
    #For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if(prediction[0]==0):
        r="SURVIVE"
    else:
        r="NOT SURVIVE"
        
    return render_template('index.html', prediction_text='YOU WILL {}'.format(r))

@new.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    #For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    new.run(debug=True)
