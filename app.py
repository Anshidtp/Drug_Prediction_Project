import joblib
from flask import Flask, request, app,jsonify, url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load The model
#drug_model = joblib.load(open('Notebook\Drug_Model .pkl', 'rb'))

@app.route('/',methods=['GET'])  # route to display the home page
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST']) 
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user

            gender=request.form['gender']
            bp = request.form['bp']
            cholesterol =request.form['cholesterol']
            age = request.form['age']
            natok = request.form['natok']
            inPut = np.array([[gender,bp,cholesterol,age,natok]])
            print(inPut)
            encoder=joblib.load(open('Notebook\Encoder.pkl','rb'))
            model=joblib.load(open('Notebook\Drug_model.pkl','rb'))
            # predictions using the loaded model file
            encoded_data = encoder.transform(inPut)
            # print(encoded_data)
            output =model.predict(encoded_data)
            print(output)
            return render_template('results.html',prediction=output[0])
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')

if __name__ =="__main__":
    app.run(debug =True)


