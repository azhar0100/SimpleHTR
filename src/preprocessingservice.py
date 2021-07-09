from flask import Flask
import requests
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from preprocessor import Preprocessor
prep = Preprocessor([128,32],data_augmentation=False)

import json
import io
from PIL import Image
import numpy as np

@app.route('/setprep',methods=['POST'])
def initprep():
    input_json = request.json
    # print(input_json)
    # rd = json.loads(str(input_json))
    rd = input_json
    args = rd['args']
    kwargs = rd['kwargs']
    prep = Preprocessor(*args,**kwargs)
    return f"Okay, {args} and {kwargs} were sent"

@app.route('/processimage',methods=["POST"])
def processimage():
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        b = io.BytesIO()
        file.save(b)
        img = Image.open(b).convert('L')
        img = np.array(img)
        print(img)
        processedimg = prep.process_img(img)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        r = requests.post('http://infero:5001/inferimage',json={'img':(processedimg.tolist()),'shape':processedimg.shape})
        # print(r.text)
        return r.text
    return
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


