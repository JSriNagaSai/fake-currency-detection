import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import cv2
from tensorflow import keras
#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.preprocessing import image


#from keras.applications import Sequential

app = Flask(__name__)
#model=keras.model.loads_model("assets/se1.h5")
#model=keras.model.load_weights('se1.h5')
model=keras.Sequential()
#model.load_weights('se1.h5')
model=load_model('se1.h5')
#model=load_model('se1.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''


    img=request.files['image']
    img.save("img.jpg")
    #img=cv2.imread("img.jpg")
    #img=resize(img,(224,224))
    img=image.load_img('img.jpg',target_size=(224,224))
    #img=image.load_img('/content/drive/MyDrive/Testdatasetall/fake/65.jpeg',target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    r=model.predict(x)
    if r==0:
        output="fake"
    else:
        output="real"

    return render_template('index.html', prediction_text='Currency is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
