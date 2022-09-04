import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, SpatialDropout2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Input, ZeroPadding2D
from tensorflow.keras import optimizers

from utils.utils import load_mel


class SoundClassificationModel(tf.keras.Model):

    def __init__(self, input_shape, classes, pretrained_weights, N_unfreezed_layers):
        super().__init__()

        self.CNN_input_shape = input_shape
        self.pretrained_weights = pretrained_weights
        self.classes = classes

        self.base_model = tf.keras.applications.MobileNetV2(
                            input_shape=self.CNN_input_shape,
                            include_top=False,
                            weights=self.pretrained_weights,
                            input_tensor=tf.keras.Input(shape=self.CNN_input_shape))
        self.base_model.trainable = False
        self.base_model = self.__make_trainable(self.base_model, N_unfreezed_layers)

        self.flatten = layers.Flatten(name="flatten")
        self.dense1 = layers.Dense(512, activation="relu")
        self.dense2 = layers.Dense(256, activation="relu")
        self.dense3 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.dense4 = layers.Dense(2, activation="softmax")
        
    # Forward pass of model - order does matter.
    def call(self, inputs, training=False):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense4(x)

    def __make_trainable(self, model, N_top_layers):
        """ Размораживает N_top_layers слоев, близких к head-у модели"""
        for layer in model.layers[len(model.layers) - N_top_layers:]:
            layer.trainable=True

        return model

    def predict_single_mel_npy_file(self, npy_path):
        """ Сделать инференс на 1 файле """
        image_arr = np.expand_dims(load_mel(npy_path), axis=-1)
        image_arr = np.dstack([image_arr for _ in range(self.CNN_input_shape[-1])])
        image_arr = tf.image.resize(image_arr, self.CNN_input_shape[:-1]).numpy()
        image_arr = tf.keras.utils.normalize(image_arr)
        class_indx = np.argmax(self.predict(image_arr[tf.newaxis]), axis=1)[0]
        return self.classes[class_indx]


class SoundDenoisingModel(tf.keras.Model):

    def __init__(self, l2_strength : float, numFeatures : int, numSegments : int):
        super().__init__()
        self.numSegments = numSegments
        self.l2_strength = l2_strength

        self.model = tf.keras.Sequential()
        self.model.add(Input(shape=[numFeatures, numSegments, 1]))
        self.model.add(ZeroPadding2D(((4,4), (0,0))))

        skip0=self._add_dense_block((9,8), first_padding="valid")
        skip1=self._add_dense_block((9,1), first_padding="same")
        self._add_dense_block((9,1), first_padding="same")
        self._add_dense_block((9,1), first_padding="same", skip_connection=skip1)
        self._add_dense_block((9,1), first_padding="same", skip_connection=skip0)

        self.model.add(SpatialDropout2D(0.2))
        self.model.add(Conv2D(filters=1, kernel_size=[129,1], strides=[1, 1], padding='same'))
        

    def _add_dense_block(self, first_Conv2D_kernel_size, first_padding, skip_connection=None):
        self.model.add(Conv2D(filters=18, kernel_size=first_Conv2D_kernel_size, strides=[1, 1], 
                       padding=first_padding, use_bias=False, 
                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_strength)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], 
                       padding='same', use_bias=False,
                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_strength)))
        middle_conv2d_outputs = self.model.outputs
        if skip_connection is not None:
            self.model.outputs += middle_conv2d_outputs
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1],
                       padding='same', use_bias=False,
                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_strength)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        return middle_conv2d_outputs

    # Forward pass of model - order does matter.
    def call(self, inputs):
        return self.model(inputs)

    def predict_unnoised_mel_npy_file(self, noisy_mel_npy_path : str):
        """ Сделать инференс на 1 файле. Вернет отфильтрованную mel-спектрограмму """
        mel_noisy = load_mel(filepth=noisy_mel_npy_path)

        mel_image_segmented = []
        for i in range(0, len(mel_noisy)-self.numSegments):
            segment = np.expand_dims(mel_noisy[i:i+self.numSegments].T, axis=-1)
            mel_image_segmented.append(segment)

        mel_image_segmented = tf.convert_to_tensor(mel_image_segmented)
        mel_filtered=np.squeeze(self.model.predict(mel_image_segmented))
        return mel_filtered
