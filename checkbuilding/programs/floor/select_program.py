from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('./uni_predict.json').read())
#保存した重みの読み込み
model.load_weights('./uni_predict.hdf5')

categories = ["2", "9", "10"]

#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

#予測結果によって処理を分ける
if features[0,0] == 1:
    print("2号間")

elif features[0,1] ==1:
    print("9号間")

else:
    print("10号間")
