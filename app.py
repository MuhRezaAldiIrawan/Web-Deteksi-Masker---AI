from flask import Flask
from flask import render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images/'
model = load_model('masker.h5')

class_dict = { 0 :'with mask' , 1 :'without mask' }



@app.route("/", methods=['GET','POST'])
def index():
	if request.method == 'POST':
		if request.files:
			image = request.files['image']
			img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
			image.save(img_path)
			prediction = get_output(img_path)
			return render_template('index.html', uploaded_image=image.filename, prediction=prediction)
	return render_template('index.html')


def get_output(img_path):
	loaded_img = load_img(img_path, target_size=(150,150))
	img_array = img_to_array(loaded_img) / 255.0
	img_array = expand_dims(img_array, 0) 
	predicted_bit = np.round(model.predict(img_array)[0][0].astype(int))
	return class_dict[predicted_bit]

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ =='__main__':
	# app.debug = True
	# app.run(debug=True)
	app.run(port=1300,debug = True)