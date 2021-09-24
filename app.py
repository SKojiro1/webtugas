from flask import Flask, render_template, request, url_for

#import io
import numpy as np
#from flask import Response
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os, time, glob
import prediction as prd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',hasil=0)

@app.route('/data')
def contoh():
    return render_template('data.txt')

@app.route('/predict', methods=['POST'])
def predict():
    result=plot_png()
    prediksi=prediction()[0]
    percen=prediction()[1]
    return render_template('home.html', hasil=getdata(), result=result, prediksi=prediksi, percen=percen) 

def prediction():
    return prd.prediction(getdata())
    
def plot_png():
    data1=getdata()
    t=np.arange(0,len(data1),1)
    u=data1
    plt.figure(figsize=(20,8))  # needed to avoid adding curves in plot
    plt.plot(t, u)
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    plotfile = os.path.join('static', str(time.time()) + '.png')  
    plt.savefig(plotfile)
    return plotfile      
    
def getdata():
    # define min max scaler
    data1=str(request.form['var1'])
    data1=data1.strip()
    data1=data1.replace(' ',',')
    data1=data1.replace('\n',',')
    data1=data1.split(',')
    data1=np.array(data1)
    data1=data1.astype(np.float_)
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data1.reshape(-1,1))
    return data_norm


if __name__ == "__main__":
    app.run(debug=True)
 

 
