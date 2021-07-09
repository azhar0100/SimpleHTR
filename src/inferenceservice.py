from model import Model, DecoderType
from dataloader_iam import DataLoaderIAM, Batch





from flask import Flask

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

import json
import io
from PIL import Image
import numpy as np
from main import FilePaths

model = Model(list(open(FilePaths.fn_char_list).read()), DecoderType.BestPath, must_restore=True)

@app.route('/inferimage',methods=["POST"])
def inferimage():
    input_json = request.json
    # print(input_json)
    img = np.array(list(input_json['img'])).reshape(input_json['shape'])
    print(img.shape)
    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return {"Recognized":str(recognized[0]), "Probability":float(probability[0])}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
