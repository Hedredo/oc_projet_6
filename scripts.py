from tensorflow.keras.models import Model
import cv2
import numpy as np
from IPython.display import clear_output


def extract_embeddings(model, images, path, preprocess_input):
    input_shape = model.input.shape
    input_size = (input_shape[1], input_shape[2])
    print(f"Input size: {input_size}")
    last_layer = model.layers[-2]
    model = Model(inputs=model.inputs, outputs=last_layer.output)
    images_embeddings = []
    for img in images:
        clear_output()
        image = cv2.imread(path + img)
        image = cv2.resize(image, input_size)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        image_embedding = model.predict(image)
        images_embeddings.append(image_embedding)
    images_embeddings = np.asarray(images_embeddings)
    return images_embeddings
