import tensorflow as tf
import numpy as np
from utils.utils import load_mel
import os

ROOT = os.path.dirname(os.path.realpath(__file__))


def predict_single_mel_npy_file(npy_path, model, classes=("clean", "noisy")):
    """ Сделать инференс на 1 файле """
    input_shape = model.layers[0].input_shape[1:]

    image_arr = np.expand_dims(load_mel(npy_path), axis=-1)
    image_arr = np.dstack([image_arr for _ in range(input_shape[-1])])
    image_arr = tf.image.resize(image_arr, input_shape[:-1]).numpy()
    image_arr = tf.keras.utils.normalize(image_arr)
    class_indx = np.argmax(model.predict(image_arr[tf.newaxis]), axis=1)[0]
    return classes[class_indx]


if __name__ == "__main__":
    # ============= Эту часть можно менять ===============================
    filepaths = ["task2/data/val/clean/264/264_121332_264-121332-0017.npy",
                 "task2/data/val/clean/614/614_6500_614-6500-0012.npy",
                 "task2/data/val/noisy/273/273_123248_273-123248-0014.npy",
                 "task2/data/val/noisy/1154/1154_129975_1154-129975-0005.npy"]

    trained_model_filepath = f"{ROOT}/pretrained_models/classification_clean_noisy"
    # ====================================================================

    model = tf.keras.models.load_model(trained_model_filepath)

    print("\n output:")
    for pth in filepaths:
        print(pth, ":", predict_single_mel_npy_file(pth, model))