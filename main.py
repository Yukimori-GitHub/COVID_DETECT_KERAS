import numpy as np
from keras.preprocessing import image


def img_prediction(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_arrary = image.img_to_array(img)
    img_arrary = np.expand_dims(img_arrary, axis=0)
    prediction = model.predict(img_arrary)
    if prediction == 0:
        print('Covid')
    else:
        print('Normal')
