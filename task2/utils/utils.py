import numpy as np
import matplotlib.pyplot as plt
import librosa

import soundfile


def show_mel_spectra(mel_img):
    plt.figure(figsize=(20, 6))
    mel_img = (mel_img-mel_img.mean()) / mel_img.std()
    plt.imshow(mel_img.astype(np.float64).T)
    print(mel_img.mean())


def load_mel(filepth):
    mel_spec=np.load(filepth).astype(np.float64)
    return mel_spec


def reconstruct_audio_from_mel(mel_spec,
                               out='rec.flac',
                               sr=16000,
                               n_fft=1024,
                               hop_length=256,
                               fmin=20,
                               fmax=8000):

    print(f"- writing {out} started")
    mel_spec = np.exp((mel_spec - 1)*10).T
    y_inv = librosa.feature.inverse.mel_to_audio(M=mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    soundfile.write(out, y_inv, samplerate=sr)
    print(f"- writing {out} finished")

