
import os
import warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ['TF_DISABLE_LAYOUT_OPTIMIZER'] = '1'
#1 geforce
#0 titan
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# Suppress TensorFlow warnings including NUMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Configure TensorFlow threading BEFORE importing TensorFlow
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.keras.backend.clear_session()

# Suppress TensorFlow logging and NUMA warnings
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure GPU memory growth to avoid NUMA warnings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration warning: {e}")

# Mixed precision disabled for numerical stability in CTC loss
# If you need mixed precision, ensure CTC loss uses float32

# Suppress NUMA warnings specifically
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', '.*NUMA.*')

tf.config.threading.set_intra_op_parallelism_threads(16)  # Half of 32 threads
tf.config.threading.set_inter_op_parallelism_threads(16)  # Half of 32 threads

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import codecs
import random
import cv2
from PIL import Image
import time
import argparse
import sys
from glob import glob
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import UpSampling2D, Concatenate, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from data import preproc as pp

import math
import tensorflow as tf
import gc
import os

rootPath = './'
DatabasePath = 'datasets/nan_raw_biner/'
max_text_length = 128
input_size_crnn = (1024, 128, 1)
def read_file_char(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())

	return list0

charset_base = read_file_char(rootPath+ 'Sets/CHAR_LIST')
print(f"Number of classes (len(charset_base) + 1): {len(charset_base) + 1}")

def read_file_shuffle(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())
	random.shuffle(list0)
	return list0
def read_file(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())

	return list0

def normalizeTranscription(text_line):
	text_line = text_line.replace('sp', ' sp ')
	text_line = text_line.replace('A', 'A ')
	text_line = text_line.replace('B', 'B ')
	text_line = text_line.replace('E', 'E ')
	text_line = text_line.replace('M', 'M ')
	text_line = text_line.replace('  ', ' ')
	return  text_line

def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		try:
			index = charset_base.index(item.lower())
			encoded.append(index + 1)
		except ValueError:
			# Handle cases where a word is not in the charset, even after converting to lowercase
			# For example, due to punctuation or special characters not in CHAR_LIST
			# print(f"Warning: Word '{item}' not found in charset. Skipping.")
			pass
		
	# encoded=encoded[::-1]  ############this is done only for arabic, otherwise remove this line

	return encoded

def readGrayPair(im_name, split='train'):
    # Path ke gambar terdegradasi dan ground truth
    deg_image_path = os.path.join('datasets/nan_distorted/', split, im_name)
    gt_image_path = os.path.join(DatabasePath, split, 'images', im_name)

    # Fungsi helper untuk memuat, mengubah ukuran, dan menormalisasi gambar
    def process_image(path):
        try:
            # Buka gambar dan konversi ke grayscale
            img = Image.open(path).convert('L').resize((1024, 128), Image.LANCZOS)
            # Konversi ke numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Pengecekan Kritis: Pastikan array tidak kosong dan berisi nilai finite
            if img_array.size == 0 or not np.all(np.isfinite(img_array)):
                print(f"⚠️  Peringatan: Data gambar tidak valid atau korup di {path}. Melewati.")
                return None

            # Normalisasi ke rentang [-1, 1]
            # img_array = (img_array / 127.5) - 1.0
            
            # Pengecekan Kritis: Pastikan normalisasi tidak menghasilkan NaN
            if np.isnan(img_array).any():
                print(f"⚠️  Peringatan: Nilai NaN setelah normalisasi di {path}. Melewati.")
                return None

            # Tambahkan dimensi channel
            return img_array[..., np.newaxis]
        except Exception as e:
            print(f"❌ Error saat memproses gambar {path}: {e}")
            return None

    # Proses kedua gambar
    deg_image = process_image(deg_image_path)
    gt_image = process_image(gt_image_path)
    
    # Jika salah satu gambar gagal diproses, kembalikan None
    if deg_image is None or gt_image is None:
        return None, None
    
    return deg_image, gt_image

class UltraSafeCTCLossLocal:
    """
    A more robust CTC loss function with extensive safety checks to prevent NaN values.
    """
    def __init__(self):
        # A high fallback loss encourages the model to correct its course if it enters an unstable state.
        self.fallback_loss = 50.0

    def safe_ctc_loss(self, y_true, y_pred):
        """
        Safely computes CTC loss with checks for NaNs, Infs, and invalid input shapes.
        """
        # 1. Squeeze and cast inputs to the correct data types.
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 2. Check for non-finite values in predictions (a sign of model instability).
        if not tf.reduce_all(tf.math.is_finite(y_pred)):
            tf.print("Debug CTC: y_pred contains non-finite values (NaN or Inf).", output_stream=sys.stderr)
            return tf.constant(self.fallback_loss, dtype=tf.float32)

        # 4. Calculate input and label lengths for CTC.
        batch_size = tf.shape(y_pred)[0]
        logit_length = tf.fill([batch_size], tf.shape(y_pred)[1])
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)

        # Ensure label_length is not greater than logit_length
        label_length = tf.minimum(label_length, logit_length)

        # --- Extensive Debugging ---
        tf.print("\n--- CTC Loss Debug Info ---", output_stream=sys.stderr)
        tf.print("y_true (shape, type, values):", tf.shape(y_true), y_true.dtype, y_true, summarize=-1, output_stream=sys.stderr)
        tf.print("y_pred (shape, type, min, max, mean):", tf.shape(y_pred), y_pred.dtype, tf.reduce_min(y_pred), tf.reduce_max(y_pred), tf.reduce_mean(y_pred), output_stream=sys.stderr)
        tf.print("label_length (shape, values):", tf.shape(label_length), label_length, summarize=-1, output_stream=sys.stderr)
        tf.print("logit_length (shape, values):", tf.shape(logit_length), logit_length, summarize=-1, output_stream=sys.stderr)
        
        # Check if any label_length is zero, which is invalid for ctc_loss
        if tf.reduce_any(tf.equal(label_length, 0)):
            tf.print("Debug CTC: Found label_length of 0. This is invalid.", output_stream=sys.stderr)
            # return tf.constant(self.fallback_loss, dtype=tf.float32)

        # Check if label_length > logit_length
        if tf.reduce_any(tf.greater(label_length, logit_length)):
            tf.print("Debug CTC: label_length > logit_length. This is invalid.", output_stream=sys.stderr)
            tf.print("Violating label_lengths:", label_length, output_stream=sys.stderr)
            tf.print("Violating logit_lengths:", logit_length, output_stream=sys.stderr)
            # return tf.constant(self.fallback_loss, dtype=tf.float32)


        try:
            # 6. Compute the CTC loss.
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=y_pred,
                label_length=label_length,
                logit_length=logit_length,
                blank_index=0, # blank_index is 0 because labels are 1-based
                logits_time_major=False,
            )

            # 7. Handle potential Inf/NaN values from the loss function itself and clip the result.
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(self.fallback_loss, dtype=tf.float32))
            loss = tf.clip_by_value(loss, 0.0, self.fallback_loss) # Clip to a max value
            
            tf.print("CTC Loss (raw):", loss, summarize=-1, output_stream=sys.stderr)
            tf.print("CTC Loss (mean):", tf.reduce_mean(loss), output_stream=sys.stderr)
            tf.print("--- End CTC Loss Debug Info ---", output_stream=sys.stderr)

            return tf.reduce_mean(loss)

        except Exception as e:
            tf.print(f"Debug CTC: Exception during ctc_loss calculation: {e}", output_stream=sys.stderr)
            return tf.constant(self.fallback_loss, dtype=tf.float32)

# Instantiate the ultra-safe CTC loss function to be used across the script
safe_ctc_loss = UltraSafeCTCLossLocal()

def flor_simplified(input_size, d_model):
    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    shape = cnn.shape
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True))(bgru)
    output_data = Dense(units=d_model, activation="linear", bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(bgru)

    return (input_data, output_data)


def build_discriminator_2():


	############################# Model Creation########################################
	# from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor_simplified(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=safe_ctc_loss.safe_ctc_loss)

	 
	return model

def data_generator(image_list, lines_list, split='train', num_samples=-1):
	"""Optimized generator function with better error handling"""
	processed_count = 0
	for im_base in image_list:
		if num_samples > 0 and processed_count >= num_samples:
			break
			
		# Find the full filename with extension
		search_pattern = os.path.join('datasets/nan_distorted/', split, im_base + '.*')
		found_files = glob(search_pattern)
		
		if not found_files:
			continue
		
		im_full_name = os.path.basename(found_files[0])
		
		# Find transcription with better error handling
		try:
			line_text = next(s for s in lines_list if s.startswith(im_full_name))
			parts = line_text.split(' ', 1)
			if len(parts) != 2:
				continue
			text_line = parts[1]
		except StopIteration:
			continue
		
		# Prepare transcription with length limits
		line = normalizeTranscription(text_line)
		words = line.split()
		if len(words) >= 20:  # Reduce max length for stability
			continue
			
		# Encode text with better error handling
		encoded_txt = encode_txt(line)
		if not encoded_txt or len(encoded_txt) > 50:  # Skip very long sequences
			continue
		
		# Ensure encoded_txt doesn't exceed max_text_length
		encoded_txt = encoded_txt[:20]  # Reduce max length
		
		try:
			# Load and preprocess images with error handling
			deg_image, gt_image = readGrayPair(im_full_name, split=split)

			# PERBAIKAN: Lewati sampel jika gambar tidak valid
			if deg_image is None or gt_image is None:
				continue
			
			# Prepare CRNN data
			gt_path = os.path.join(DatabasePath, split, 'images', im_full_name)
			img = pp.preprocess(gt_path, input_size_crnn)

			# --- Debugging Data --- 
			if img is None:
				print(f"Skipping blank image: {im_full_name}")
				continue
			print(f"Image: {im_full_name}, Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")
			
			# Transpose img for CRNN
			if len(img.shape) == 2:
				img = img.T
				img = img[..., np.newaxis]
			elif len(img.shape) == 3 and img.shape == (128, 1024, 1):
				img = np.transpose(img, (1, 0, 2))
			
			# Pad encoded_txt to fixed length
			padded_encoded = np.zeros(max_text_length, dtype=np.int16)
			padded_encoded[:len(encoded_txt)] = encoded_txt
			
			processed_count += 1
			
			yield img.astype(np.float32), padded_encoded
			
		except Exception as e:
			print(f"⚠️ Error processing {im_full_name}: {e}")
			continue

def create_optimized_dataset(list_image_train, list_lines, split, strategy, batch_size=12, num_samples=-1):
	"""Create highly optimized dataset pipeline with aggressive optimizations"""
	
	AUTOTUNE = tf.data.AUTOTUNE
	
	# Create base dataset with smaller buffer for faster iteration
	dataset = tf.data.Dataset.from_generator(
		lambda: data_generator(list_image_train, list_lines, split, num_samples=num_samples),
		output_signature=(tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32), tf.TensorSpec(shape=(max_text_length,), dtype=tf.int16))
	)
	
	per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
	# Aggressive optimizations for speed
	if num_samples > 0:
		dataset = dataset.take(num_samples)
		print(f"🚀 OPTIMIZED Dataset: taking {num_samples} samples, batch={per_replica_batch_size}")
	else:
		print(f"🚀 OPTIMIZED Dataset: using all available samples, batch={per_replica_batch_size}")

	dataset = dataset.cache()  # Cache in memory
	dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)  # Smaller buffer
    
	# Faster parallel processing
	dataset = dataset.map(
		lambda x, y: (x, y),
		num_parallel_calls=8,  # Reduced from AUTOTUNE for stability
		deterministic=False
	)
	
	# Batch optimization
	dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
	
	# Reduced prefetch for faster iteration
	dataset = dataset.prefetch(2)
	
	return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')
    list_lines = read_file(rootPath + 'Sets/lines.txt')
    list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_discriminator_2()

    train_dataset = create_optimized_dataset(list_image_train, list_lines, 'train', strategy, batch_size=args.batch_size, num_samples=50)
    valid_dataset = create_optimized_dataset(list_image_valid, list_lines, 'validation', strategy, batch_size=args.batch_size, num_samples=50)

    # Create an intermediate model to inspect the output of the GRU layer
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(name='bidirectional').output)

    for i, (images, labels) in enumerate(train_dataset.take(1)):
        intermediate_output = intermediate_model.predict(images)
        print(f"\n--- Intermediate Layer (Bidirectional GRU) Output Debug Info (Batch {i+1}) ---")
        print(f"Shape: {intermediate_output.shape}")
        print(f"Min: {np.min(intermediate_output)}")
        print(f"Max: {np.max(intermediate_output)}")
        print(f"Mean: {np.mean(intermediate_output)}")
        print("--- End Intermediate Layer Output Debug Info ---")

    # Inspect the weights of the final Dense layer
    dense_layer = model.get_layer(name='dense')
    weights, biases = dense_layer.get_weights()

    print(f"\n--- Final Dense Layer Weights Debug Info ---")
    print(f"Weights Shape: {weights.shape}")
    print(f"Weights Min: {np.min(weights)}")
    print(f"Weights Max: {np.max(weights)}")
    print(f"Weights Mean: {np.mean(weights)}")
    print(f"Biases Shape: {biases.shape}")
    print(f"Biases Min: {np.min(biases)}")
    print(f"Biases Max: {np.max(biases)}")
    print(f"Biases Mean: {np.mean(biases)}")
    print("--- End Final Dense Layer Weights Debug Info ---")

    model.fit(train_dataset, epochs=args.epochs, validation_data=valid_dataset)
