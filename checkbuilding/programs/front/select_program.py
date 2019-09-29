from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os, tkinter, tkinter.filedialog, tkinter.messagebox

model = model_from_json(open('../../data/front/uni_predict.json').read())
model.load_weights('../../data/front/uni_predict.hdf5')

categories = ["2", "9", "10"]

while True:
    root = tkinter.Tk()
    root.withdraw()
    iDir = '../../images/front'
    file = tkinter.filedialog.askopenfilename(initialdir=iDir)

    img = image.load_img(file, target_size=(150, 150, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    features = model.predict(x)

    if features[0, 0] == 1:
        tkinter.messagebox.showinfo('判定', '2号館')

    elif features[0, 1] == 1:
        tkinter.messagebox.showinfo('判定', '9号館')

    else:
        tkinter.messagebox.showinfo('判定', '10号館')
