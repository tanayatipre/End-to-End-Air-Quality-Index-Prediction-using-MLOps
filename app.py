from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from MLProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")