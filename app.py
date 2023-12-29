import numpy as np
import pandas as pd
import scipy
import joblib
import os
import gc
from sklearn.metrics import label_ranking_average_precision_score
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback 
from tensorflow.keras.layers import Flatten, Input, Dense
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2
import streamlit as st

# Reference implementation from: https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8
def _lwlrap_sklearn(truth, scores):
    '''
    Description -> Return the Label Weighted Label Ranking Average Precision (LWLRAP) of the given true and predicted
    class labels
    
    Inputs ->
    truth: OHE Vector of truth class label
    scores: NumPy array of predictions
    
    Output -> LWLRAP score
    '''
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0, 
        scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def _one_sample_positive_class_precisions(example):
    y_true, y_pred = example
    retrieved_classes = tf.argsort(y_pred, direction='DESCENDING')
    class_rankings = tf.argsort(retrieved_classes)
    retrieved_class_true = tf.gather(y_true, retrieved_classes)
    retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))

    idx = tf.where(y_true)[:, 0]
    i = tf.boolean_mask(class_rankings, y_true)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = 1 + tf.cast(i, tf.float32)
    precisions = r / c
    dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
    return dense

class LWLRAP(Metric):
    def __init__(self, num_classes, name='lwlrap'):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros')

        self._counts = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=_one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32))

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)        

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)


class Config():
    def __init__(self, sampling_rate, n_classes=80):
        self.sampling_rate=sampling_rate
        self.n_classes=n_classes
        self.stft_window_seconds=0.025
        self.stft_hop_seconds=0.010
        self.mel_bands=96
        self.mel_min_hz=20
        self.mel_max_hz=20000
        self.mel_log_offset=0.001
        self.example_window_seconds=1.0
        self.example_hop_seconds=0.5

def fetch_map(train_csv_path=None):
    """
    Creates a hash table mapping between each integer from 0-79 and each unique class label.
    Args:
        train_csv_path (str): Path to the "train_curated.csv" file which is used to obtain the list of all class labels.
    Returns:
        dict: The class mapping hash table where the key is an integer (0-79) and the value is the corresponding
        class label.
    """
    # Read the input csv file
    df_train = pd.read_csv(train_csv_path)

    # Create a set of all unique class labels present in the above file
    unique_labels = set(df_train['labels'].str.split(',').explode().unique())

    # Sort the set in alphabetical order
    sorted_labels = sorted(unique_labels)

    # Create the hash table
    class_map = {i: label for i, label in enumerate(sorted_labels)}

    return class_map

def process(clip, clip_dir=None):
    """Decodes a WAV clip into a batch of log mel spectrum examples.

    This function takes the given .wav file, gets its tensor representation, converts it into spectrogram using short-time
    Fourier transform, then converts the spectrogram into log mel spectrogram, finally, it divides it into various windows
    and returns all the windows in a 3-channel format.

    Args:
        clip (str): Path to .wav file, e.g., 'file1.wav'.
        clip_dir (str, optional): Parent folder in which the above clips is stored, e.g., 'preprocessed_dir'.

    Returns:
        tf.Tensor: Log mel spectrogram windowed features.
    """
    # Decode WAV clip into waveform tensor.   
    form_wave = tf.squeeze(tf.audio.decode_wav(tf.io.read_file(clip))[0])

    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    window_length_samples = int(round(config.sampling_rate * config.stft_window_seconds))
    hop_length_samples = int(round(config.sampling_rate * config.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log2(window_length_samples)))
    
    magnitude_spectrogram = tf.math.abs(tf.signal.stft(signals=form_wave,
                                                       frame_length=window_length_samples,
                                                       frame_step=hop_length_samples,
                                                       fft_length=fft_length))

    # Convert spectrogram into log mel spectrogram.
    num_spectrogram_bins = fft_length // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=config.mel_bands,
                                                                        num_spectrogram_bins=num_spectrogram_bins,
                                                                        sample_rate=config.sampling_rate,
                                                                        lower_edge_hertz=config.mel_min_hz,
                                                                        upper_edge_hertz=config.mel_max_hz)
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + config.mel_log_offset)

    # Frame log mel spectrogram into examples.
    spectrogram_sr = 1 / config.stft_hop_seconds
    example_window_length_samples = int(round(spectrogram_sr * config.example_window_seconds))
    example_hop_length_samples = int(round(spectrogram_sr * config.example_hop_seconds))
    features = tf.signal.frame(signal=log_mel_spectrogram,
                               frame_length=example_window_length_samples,
                               frame_step=example_hop_length_samples,
                               pad_end=True,
                               pad_value=0.0,
                               axis=0)
    
    # Converting mono channel to 3 channels
    features = mono_to_color(features)
    features=tf.stack([features,features,features], axis=-1)      
    return features

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    """
    Description - The mono_to_color function converts a grayscale image to a colored image. It applies standardization to the input 
                  data and then normalizes it between the values of norm_min and norm_max. If the difference between the minimum and
                  maximum values is greater than eps, it then maps the normalized values to the range [0, 255] to obtain a colored 
                  image. If the difference is smaller than eps, the function returns a tensor of zeros with the same shape as the 
                  input tensor. If the mean and std parameters are not provided, the function calculates them from the input tensor.
                  If norm_min and norm_max are not provided, the function calculates them from the normalized input tensor. 
                  The eps parameter is used to avoid division by zero.
    """    
    # Standardize
    mean = mean or tf.math.reduce_mean(X)
    std = std or tf.math.reduce_std(X)
    Xstd = (X - mean) / (std + eps)
    _min, _max = tf.math.reduce_min(Xstd), tf.math.reduce_max(Xstd)
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V = tf.where(V < norm_min, norm_min, V)
        V = tf.where(V > norm_max, norm_max, V)
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = tf.cast(V, tf.float32)
    else:
        # Just zero
        V = tf.zeros_like(Xstd, dtype=tf.float32)
    return V   

@st.cache_resource
def cnnmodel(weights_path=None):
    '''
    Description - This function returns a 2D CNN model. 
    If a "weights_path" is provided, it returns the model with the best weights for testing. 
    If not, it returns the compiled model for training.
    '''    
    model = EfficientNetV2B2(include_top=False, input_shape=(100, 96, 3))
    x = Flatten()(model.layers[-1].output)
    out = Dense(80)(x)
    model = Model(inputs=model.input, outputs=out)
    if not weights_path: 
        model.compile(optimizer='adam',
                      loss=tf.nn.sigmoid_cross_entropy_with_logits,
                      metrics=[LWLRAP(80)])
    else:
        model.load_weights(weights_path)        
    return model

config = Config(44100)

def main():
  """
  Description -> Given a raw audio signal X, this function processes the audio signal into features, 
                 feeds it to a pre-trained convolutional neural network model, and generates a pandas DataFrame 
                 containing the top 5 predicted labels and their associated probabilities, sorted in descending order. 

                  Args:
                  - X: A raw audio signal in the form of a numpy array. 

                  Returns:
                  - A pandas DataFrame containing the top 5 predicted labels and their associated probabilities, 
                    sorted in descending order.
  """  
  st.title('Audio Classification App')
  uploaded_file = st.file_uploader("Upload Audio File", type=['wav'])

  if uploaded_file is not None:
      audio_bytes = uploaded_file.read()
      st.audio(audio_bytes, format='audio/wav')

      if st.button('Predict'):
        features = process(X)
        model = cnnmodel(r"weights1_8-loss_0.0024_lwlrap_0.9922.h5")
        prediction = np.average((1/(1+np.exp(-model.predict(features)))),axis=0)
        prediction_sorted = np.argsort(prediction)
        labmap = fetch_map(r'train_curated.csv')
        topfive = [labmap[i] for i in prediction_sorted[-5:][::-1]]
        topfiveprob = prediction[prediction_sorted[-5:][::-1]]        
        result = pd.DataFrame({topfive[i]:topfiveprob[i] for i in range(5)},index=[0])
        st.markdown(result.to_markdown())

if __name__=="__main__":
  main()
