import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
app = Flask(__name__)
#my_dir=os.path.dirname(__file__)
#my_file = os.path.join(my_dir,'Cleaned_data.csv')
data = pd.read_csv('Cleaned_data.csv')
#my_dir2=os.path.dirname(__file__)
#my_file2 = os.path.join(my_dir2,'RidgeModel.pkl')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))
@app.route('/')
def index():
           locations = sorted(data['location'].unique())
           return render_template('index.html', locations=locations)
@app.route('/predict', methods=['POST'])
def Predict():
          location = request.form.get('location')
          bhk =  (request.form.get('bhk'))
          bath = (request.form.get('bath'))
          sqft =  (request.form.get('total_sqft'))

          df = pd.DataFrame([[location,float(sqft),float(bath),int(bhk)]],columns=['location', 'total_sqft', 'bath', 'bhk'])
          prediction = pipe.predict(df)[0]*1e5
          return str(np.round(prediction,2))


if __name__== "__main__":
  app.run(debug=True, port=5001)
