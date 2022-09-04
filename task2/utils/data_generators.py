import random
import numpy as np

from glob import glob
import tensorflow as tf


class ClassificationDataGenerator(tf.keras.utils.Sequence):
    """ Генератор данных для подгрузки данных при тренировке и валидации нейросети.
        В качестве кодировки разметки используется one-hot encoding. """
    
    def __init__(self, data_folders : list,
                 batch_size : int,
                 input_image_size : tuple,
                 image_classes : tuple,
                 shuffle : bool):

        self.batch_size = batch_size
        self.input_image_size = input_image_size
        self.image_classes = image_classes
        self.shuffle = shuffle

        self.file_paths = []
        for data_folder in data_folders:
            self.file_paths += glob(f"{data_folder}/clean/*/*.npy") + glob(f"{data_folder}/noisy/*/*.npy")
            random.shuffle(self.file_paths)

        self.n = len(self.file_paths)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_paths)

    def __load_image(self, path, target_size):
        image_arr = np.expand_dims(np.load(path), axis=-1)
        image_arr = np.dstack([image_arr for _ in range(target_size[-1])])
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        return tf.keras.utils.normalize(image_arr)

    def __get_image_label(self, path):
        label = self.image_classes.index(path.replace("\\", "/").split("/")[2])
        return tf.keras.utils.to_categorical(label, num_classes=len(self.image_classes))

    def __get_data(self, file_path_batches):
        X_batch = np.asarray([self.__load_image(pth, self.input_image_size) for pth in file_path_batches])
        y_batch = np.asarray([self.__get_image_label(pth) for pth in file_path_batches])
        return X_batch, y_batch

    def __getitem__(self, index):
        file_path_batches = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(file_path_batches)        
        return X, y

    def __len__(self):
        return self.n // self.batch_size


class DenoisingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_folders : list,
                 batch_size : int,
                 numSegments : int,
                 shuffle : bool):

        self.batch_size = batch_size
        self.numSegments = numSegments
        self.shuffle = shuffle

        self.file_paths = []
        for data_folder in data_folders:
            self.file_paths += list(zip(glob(f"{data_folder}/noisy/*/*.npy"), glob(f"{data_folder}/clean/*/*.npy")))
            random.shuffle(self.file_paths)

        self.n = len(self.file_paths)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_paths)

    def __load_noisy(self, path):
        """итерируем не по каждому вектору mel-спектрограмм, а через numSegments раз для уменьшения вычислительных нагрузок"""
        mel_image = np.load(path)
        mel_image_segmented = []
        for i in range(self.numSegments, len(mel_image), self.numSegments):
            segment = np.expand_dims(mel_image[i-self.numSegments:i].T, axis=-1)
            mel_image_segmented.append(segment)
        return tf.convert_to_tensor(mel_image_segmented)

    def __load_clean(self, path):
        """итерируем не по каждому вектору mel-спектрограмм, а через numSegments раз для уменьшения вычислительных нагрузок"""
        mel_image = np.load(path)
        mel_image_segmented = []
        for i in range(self.numSegments, len(mel_image), self.numSegments):
            segment = mel_image[i].T
            segment = np.expand_dims(segment, axis=-1)
            segment = np.expand_dims(segment, axis=-1)
            mel_image_segmented.append(segment)
        return tf.convert_to_tensor(mel_image_segmented)

    def __get_data(self, file_path_batches):
        X_batch = []
        y_batch = []

        for pth in file_path_batches:
            self.__load_noisy(pth[0])
            X_batch.extend(self.__load_noisy(pth[0]))
            y_batch.extend(self.__load_clean(pth[1]))

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        return X_batch, y_batch

    def __getitem__(self, index):
        file_path_batches = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(file_path_batches)        
        return X, y

    def __len__(self):
        return self.n // self.batch_size
