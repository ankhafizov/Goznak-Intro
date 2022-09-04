import os

import tensorflow as tf
import numpy as np
from scipy.ndimage import median_filter
from pathlib import Path

from utils.utils import load_mel, reconstruct_audio_from_mel

ROOT = os.path.dirname(os.path.realpath(__file__))


def predict_unnoised_mel_npy_file(noisy_mel_npy_path, model, numSegments=8):
    """ Сделать инференс на 1 файле """
    mel_noisy = load_mel(filepth=noisy_mel_npy_path)
    mel_image_segmented = []
    for i in range(0, len(mel_noisy)-numSegments):
        segment = np.expand_dims(mel_noisy[i:i+numSegments].T, axis=-1)
        mel_image_segmented.append(segment)

    mel_image_segmented = tf.convert_to_tensor(mel_image_segmented)
    mel_filtered=np.squeeze(model.predict(mel_image_segmented))
    return mel_filtered


if __name__ == "__main__":
    # ============= Эту часть можно менять ===============================
    filepaths = [f"{ROOT}/data/val/noisy/273/273_123248_273-123248-0014.npy",
                 f"{ROOT}/data/val/noisy/1154/1154_129975_1154-129975-0005.npy"]

    trained_model_filepath = f"{ROOT}/pretrained_models/denoising"
    out_folder_path = "out"
    # ====================================================================

    model = tf.keras.models.load_model(trained_model_filepath)

    for pth in filepaths:
        tag = pth.split("/")[-1].split(".")[0]
        Path(f"{ROOT}/{out_folder_path}/{tag}").mkdir(parents=True, exist_ok=True)
        noised_mel = load_mel(pth)
        unnoised_mel = predict_unnoised_mel_npy_file(pth, model)

        reconstruct_audio_from_mel(noised_mel, f"{ROOT}/{out_folder_path}/{tag}/noised.flac")
        reconstruct_audio_from_mel(unnoised_mel, f"{ROOT}/{out_folder_path}/{tag}/unnoised.flac")
        reconstruct_audio_from_mel(median_filter(unnoised_mel, footprint=np.ones((3, 1))),
                                   f"{ROOT}/{out_folder_path}/{tag}/unnoised_median.flac")

        try:
            clean_mel = load_mel(pth.replace("noisy", "clean"))
            reconstruct_audio_from_mel(clean_mel, f"{ROOT}/{out_folder_path}/{tag}/clean.flac")
        except FileNotFoundError:
            pass