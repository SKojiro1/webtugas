from flask import Flask, render_template, request, url_for

import io
import numpy as np
from flask import Response
import matplotlib.pyplot as plt

import os, time, glob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    result=plot_png()
    return render_template('home.html', hasil=getdata(), result=result) 

def plot_png():
    data1=getdata()
    t=np.arange(0,len(data1),1)
    u=data1
    plt.figure()  # needed to avoid adding curves in plot
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
    data1=str(request.form['var1'])
    data1=data1.strip()
    data1=data1.replace(' ','')
    data1=data1.replace('\n',',')
    data1=data1.split(',')
    data1=np.array(data1)
    data1=data1.astype(np.float_)
    print("size: ", np.size(data1))
    print("shape: ", data1.shape)
    return data1


if __name__ == "__main__":
    app.run(debug=True)
 
 
 
