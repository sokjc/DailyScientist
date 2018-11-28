import flask
from flask import Flask, request, render_template
#from sklearn.externals import joblib
import numpy as np

from sklearn.cluster import KMeans
import cv2
from scipy import misc
import pandas as pd
from scipy.spatial import distance as dist
from collections import OrderedDict

from webcolors import rgb_to_hex
from math import floor

app = Flask(__name__)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def rgb2hex(r,g,b):
    hex = "#{:02x}{:02x}{:02x}".format(r,g,b)
    return hex

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route("/analyze", methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        file = request.files['image']
        if not file: return render_template('index.html', label="No file")

        #convert string data to numpy array
        npimg = np.fromstring(file.read(), np.uint8)

        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        image = img.reshape((img.shape[0] * img.shape[1],3))

        clt = KMeans(n_clusters=3)
        clt.fit(image)

        hist = centroid_histogram(clt)
        hex_ = [rgb_to_hex([floor(i) for i in c]) for c in clt.cluster_centers_.tolist()]

        viz_data = {'colors':hex_, 'dist':hist.tolist()}

        return render_template('index.html', bob=hex_)

if __name__ == '__main__':
    #Start up app
    app.run(host="0.0.0.0", port=8000, debug = True)
