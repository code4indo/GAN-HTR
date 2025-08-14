import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ['TF_DISABLE_LAYOUT_OPTIMIZER'] = '1'
#1 geforce
#0 titan
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# Configure TensorFlow threading BEFORE importing TensorFlow
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(16)  # Half of 32 threads
tf.config.threading.set_inter_op_parallelism_threads(16)  # Half of 32 threads

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import codecs
import random
import cv2
import time
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

# Advanced GPU and performance configuration for dual RTX A4000
def configure_optimal_gpu_setup():
    """Configure optimal settings for dual RTX A4000 GPUs"""
    
    # Set environment variables for optimal performance
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
    
    # Configure GPU memory growth and optimization
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable TF32 for RTX series (faster training) - if available
            try:
                tf.config.experimental.enable_tensor_float_32()
                print("✅ TF32 enabled for faster training")
            except AttributeError:
                print("⚠️ TF32 not available in this TensorFlow version - skipping")
            
            print(f"Configured {len(gpus)} GPUs with optimal settings")
            print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Setup distributed strategy for multi-GPU
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"🚀 Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        return strategy
    else:
        print("⚠️  Single GPU detected, using OneDeviceStrategy")
        return tf.distribute.OneDeviceStrategy("/gpu:0")

# Configure CPU threading for Threadripper PRO 3955WX (32 threads)
def configure_cpu_optimization():
    """Configure optimal CPU settings for 32-thread Threadripper"""
    print("🔧 CPU threading already optimized for 32-thread Threadripper PRO")

# Advanced performance optimizations
def enable_advanced_optimizations():
    """Enable XLA, mixed precision, and other optimizations"""
    
    # Enable XLA JIT compilation for faster execution
    tf.config.optimizer.set_jit(True)
    
    # Enable mixed precision (already set but ensuring it's optimal)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("⚡ Advanced optimizations enabled: XLA JIT, Mixed Precision, TF32")

# Initialize all optimizations
print("🚀 Initializing Full Optimization Strategy...")
strategy = configure_optimal_gpu_setup()
configure_cpu_optimization()
enable_advanced_optimizations()
print("✅ Full optimization configuration completed!")

from PIL import Image
from tqdm import tqdm
import random
import sys
import codecs
import re
from data import preproc as pp


##########################################################################################################
##########################################################################################################
##########################################################################################################
rootPath='./'
DatabasePath='datasets/nan_raw_biner/'
scenario='S_iam_OP'

# define parameters
source = "iam"
arch = "flor" ########ne pas modifier, nous utilisons architeture crnn
batch_size=12  # Changed from 32 to 12 to match the error output
# define paths
source_path = os.path.join("..", "data", f"{source}.hdf5")
output_path = os.path.join("..", "output-crnn-gan-" + scenario  , source, arch)
target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
os.makedirs(output_path, exist_ok=True)

source_path2 = os.path.join("..", "data", f"{source}.hdf5")
output_path2 = os.path.join("..", "output-crnn-gan-progressive-" + scenario, source, arch)
target_path2 = os.path.join(output_path2, "checkpoint_weights.hdf5")
os.makedirs(output_path2, exist_ok=True)


# define input size, number max of chars per line and list of valid chars 
max_text_length = 128  ####not change this value
img_width=1024 #########for crnn
img_height=128 #########for crnn
input_size_crnn = (1024,128, 1)
input_size = (128,1024, 1) #############for the GAN
i =1 
flag = 0



##########################################################################################################
##########################################################################################################
##########################################################################################################
def get_callbacks(logdir, checkpoint, monitor="loss", verbose=1):
        """Setup the list of callbacks for the model"""

        callbacks = [

            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks

def normalizeTranscription(text_line):
	text_line = text_line.replace('sp', ' sp ')
	text_line = text_line.replace('A', 'A ')
	text_line = text_line.replace('B', 'B ')
	text_line = text_line.replace('E', 'E ')
	text_line = text_line.replace('M', 'M ')
	text_line = text_line.replace('  ', ' ')
	return  text_line

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
def read_file_char(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())

	return list0
charset_base = read_file_char(rootPath+ 'Sets/CHAR_LIST')
f=codecs.open('charlist.txt','w','utf-8')
f.writelines(charset_base)
f.close()


def unet(pretrained_weights=None, input_size=(128,1024, 1)):
	inputs = Input(input_size)


	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	bn = BatchNormalization(momentum=0.8)(conv1)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(bn)


	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	bn = BatchNormalization(momentum=0.8)(conv2)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(bn)


	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	bn = BatchNormalization(momentum=0.8)(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	bn = BatchNormalization(momentum=0.8)(conv4)
	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv4)
	drop4 = Dropout(0.5)(bn)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	bn = BatchNormalization(momentum=0.8)(conv5)
	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv5)
	drop5 = Dropout(0.5)(bn)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
# 	 merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up6)
	merge6 = concatenate ([drop4, bn])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	bn = BatchNormalization(momentum=0.8)(conv6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up7)
	merge7 = concatenate ([conv3, bn])
# 	 merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	bn = BatchNormalization(momentum=0.8)(conv7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv7)


	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up8)
	merge8 = concatenate ([conv2, bn])
# 	 merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	bn = BatchNormalization(momentum=0.8)(conv8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
# 	 merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up9)
	merge9 = concatenate ([conv1, bn])

	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(bn)

	model = Model(inputs=inputs, outputs=conv10)

# 	 model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	return model

def get_optimizer():
	return Adam(learning_rate=1e-4)


def build_discriminator_1():

	def d_layer(layer_input, filters, f_size=4, bn=True):
# 		 """Discriminator layer"""
		d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		if bn:
			d = BatchNormalization(momentum=0.8)(d)
		return d

	img_A = Input(shape=(128,1024, 1))
	img_B = Input(shape=(128,1024, 1))
	# img_C = Input(shape=(32,768, 1))
	df = 64
	# Concatenate image and conditioning image by channels to produce input
	combined_imgs = Concatenate(axis=-1)([img_A, img_B])

	d1 = d_layer(combined_imgs, df, bn=False)
	d2 = d_layer(d1, df * 2)
	d3 = d_layer(d2, df * 4)
	d4 = d_layer(d3, df * 4)

	validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

	discriminator = Model([img_A, img_B], validity)
	
	
	discriminator.compile(loss='mse', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
	return discriminator
#######################CRNN CTC Recognize##########################
def ctc_loss_lambda_func(y_true, y_pred):
    """
    Compute CTC loss using tf.nn.ctc_loss for better stability.
    """
    # Ensure y_true is of integer type for CTC loss calculation.
    y_true = tf.cast(y_true, tf.int32)

    # Get the length of the predictions (time steps).
    sequence_length = tf.shape(y_pred)[1]
    
    # Create a tensor for input_length (logit_length).
    batch_size = tf.shape(y_pred)[0]
    input_length = tf.fill([batch_size], sequence_length)

    # Calculate the length of the true labels.
    label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)

    # Use tf.nn.ctc_loss which is more direct and stable.
    # It expects logits to be time-major, so we need to transpose y_pred.
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=False, # y_pred is [batch, time, features]
        blank_index=-1 # Let TF automatically handle blank index
    )
    
    # Return the mean loss, handling potential inf values.
    loss = tf.cast(loss, tf.float32)
    return tf.reduce_mean(tf.where(tf.math.is_inf(loss), 0.0, loss))

def build_discriminator_2():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

	 
	return model 
 
def build_discriminator_3():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

	 
	return model
	
def readGrayPair(im_name, split='train'):
	deg_image_path = os.path.join('datasets/nan_distorted/', split, im_name)

	original_image = Image.open(deg_image_path)  # /255.0
	original_image = original_image.resize((1024,128), Image.LANCZOS)
	grey_image = original_image.convert('L')
	
	grey_image.save("deg_image2.jpg")
	deg_image = plt.imread("deg_image2.jpg")
	
	gt_image_path = os.path.join(DatabasePath, split, 'images', im_name)
	original_image = Image.open(gt_image_path)
	original_image = original_image.resize((1024,128), Image.LANCZOS)
	grey_image = original_image.convert('L')
	grey_image.save("gt_image2.jpg")
	gt_image = plt.imread("gt_image2.jpg")
	
	# Ensure images have channel dimension (128, 1024, 1)
	if len(deg_image.shape) == 2:
		deg_image = deg_image[..., np.newaxis]
	if len(gt_image.shape) == 2:
		gt_image = gt_image[..., np.newaxis]
	
	return deg_image, gt_image
  
def vconcat_resize(img_list, interpolation  
                   = cv2.INTER_CUBIC): 
      # take minimum width 
    w_min = min(img.shape[1]  
                for img in img_list) 
      
    # resizing images 
    im_list_resize = [cv2.resize(img, 
                      (w_min, int(img.shape[0] * w_min / img.shape[1])), 
                                 interpolation = interpolation) 
                      for img in img_list] 
    # return final image 
    return cv2.vconcat(im_list_resize) 	
 
###############New GAN######################
def get_gan_network(discriminator_1,discriminator_2, generator, optimizer):
	
	discriminator_1.trainable = False
	discriminator_2.trainable = False

	gan_input = Input(shape=(128,1024, 1))  ######### this is the degraded image because it is a cgan

	# input_length = layers.Input(shape=[1], dtype=tf.int32, name='input_length')
	# label_length = layers.Input(name='label_length', shape=[1], dtype=tf.int32)


	out_generator = generator(gan_input)
	out_discrimintor_1 = discriminator_1([out_generator, gan_input])    ### remove the gan input 3 from here 
	######################Here we should reshape out_generator to be fed to the RCNN model
	###################### The RCNN accept shape (1024,128,1)
	reshaped = Reshape((1024,128,1 ), input_shape=(128,1024,1))(out_generator)

	out_discrimintor_2= discriminator_2([reshaped])    ### remove the gan input 3 from here : CRNN Recognizer
	# define composite model
	# out_generator is to compute the BCE loss ....
	# define composite model
	gan = Model([gan_input], [out_discrimintor_1, out_generator, out_discrimintor_2])

	gan.compile(loss=['mse','binary_crossentropy',ctc_loss_lambda_func], loss_weights=[1,10,1], optimizer=optimizer)   ##### the weight are to discuss later Please dont forget !!!
	return gan



def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		try:
			index = charset_base.index(item.lower())
			encoded.append(index)
		except ValueError:
			# Handle cases where a word is not in the charset, even after converting to lowercase
			# For example, due to punctuation or special characters not in CHAR_LIST
			# print(f"Warning: Word '{item}' not found in charset. Skipping.")
			pass
		
	# encoded=encoded[::-1]  ############this is done only for arabic, otherwise remove this line

	return encoded

def data_generator(image_list, lines_list, split='train'):
	"""Generator function untuk tf.data pipeline"""
	for im_base in image_list:
		# Find the full filename with extension
		search_pattern = os.path.join('datasets/nan_distorted', split, im_base + '.*')
		found_files = glob(search_pattern)
		
		if not found_files:
			continue
		
		im_full_name = os.path.basename(found_files[0])
		
		# Find transcription
		try:
			line_text = next(s for s in lines_list if s.startswith(im_full_name))
			parts = line_text.split(' ', 1)
			if len(parts) != 2:
				continue
			text_line = parts[1]
		except StopIteration:
			continue
		
		# Prepare transcription
		line = normalizeTranscription(text_line)
		words = line.split()
		if len(words) >= max_text_length:
			continue
			
		# Encode text
		encoded_txt = encode_txt(line)
		if not encoded_txt:  # Skip if encoding failed
			continue
		
		# Ensure encoded_txt doesn't exceed max_text_length
		encoded_txt = encoded_txt[:max_text_length-1]  # Leave room for padding
		
		# Load and preprocess images
		deg_image, gt_image = readGrayPair(im_full_name, split=split)
		
		# Prepare CRNN data
		gt_path = os.path.join(DatabasePath, split, 'images', im_full_name)
		img = pp.preprocess(gt_path, input_size_crnn)
		
		# Transpose img for CRNN: (128, 1024) -> (1024, 128, 1)
		if len(img.shape) == 2:  # (128, 1024)
			img = img.T  # -> (1024, 128)
			img = img[..., np.newaxis]  # -> (1024, 128, 1)
		elif len(img.shape) == 3 and img.shape == (128, 1024, 1):  # (128, 1024, 1)
			img = np.transpose(img, (1, 0, 2))  # -> (1024, 128, 1)
		
		# Pad encoded_txt to max_text_length
		padded_encoded = np.pad(encoded_txt, (0, max_text_length - len(encoded_txt)), mode='constant')
		
		yield {
			'deg_image': deg_image.astype(np.float32),
			'gt_image': gt_image.astype(np.float32),
			'crnn_image': img.astype(np.float32),
			'transcription': padded_encoded.astype(np.int16),
			'text_line': line
		}

def create_optimized_dataset(list_image_train, list_lines, split, strategy, batch_size=12):
	"""Create highly optimized dataset pipeline for dual-GPU training"""
	
	AUTOTUNE = tf.data.AUTOTUNE
	
	# Create base dataset
	dataset = tf.data.Dataset.from_generator(
		lambda: data_generator(list_image_train, list_lines, split),
		output_signature={
			'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
			'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
			'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
			'transcription': tf.TensorSpec(shape=(max_text_length,), dtype=tf.int16),
			'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
		}
	)
	
	# Advanced pipeline optimizations
	dataset = dataset.cache()  # Cache dataset in memory (125GB RAM available)
	dataset = dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
	
	# Parallel data processing using all 32 CPU threads
	dataset = dataset.map(
		lambda x: x,  # Identity function, but enables parallel processing
		num_parallel_calls=AUTOTUNE,
		deterministic=False  # Allow reordering for performance
	)
	
	# Batch and optimize for multi-GPU
	per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
	dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
	
	# Advanced prefetching for GPU utilization
	dataset = dataset.prefetch(AUTOTUNE)
	
	print(f"📊 Dataset optimized: global_batch={batch_size}, per_replica={per_replica_batch_size}")
	print(f"🔧 Using {strategy.num_replicas_in_sync} GPUs with advanced pipeline optimizations")
	
	return dataset

def train_gan(generator, discriminator_1, discriminator_2, gan, ep_start=0, epochs=1, batch_size=12):
	"""Optimized multi-GPU GAN training with full performance optimizations"""
	
	# Prepare data lists
	list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')
	list_lines = read_file(rootPath + 'Sets/lines.txt')
	
	# Use the global strategy for distributed training
	global strategy
	
	print(f"🚀 Starting optimized training with {strategy.num_replicas_in_sync} GPUs")
	print(f"📊 Global batch size: {batch_size} (per GPU: {batch_size // strategy.num_replicas_in_sync})")
	
	# Create optimizers and define training step in strategy scope
	with strategy.scope():
		print("🔧 Creating optimizers in distributed strategy scope...")
		gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
		disc1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
		disc2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
		
		# Define distributed training step within strategy scope
		@tf.function
		def distributed_train_step(batch_data):
			"""Optimized distributed training step"""
			
			def train_step(inputs):
				batch_train = inputs['deg_image']
				batch_target = inputs['gt_image']
				x_train_rcnn = inputs['crnn_image']
				y_train_rcnn = inputs['transcription']
				
				per_replica_batch_size = tf.shape(batch_train)[0]
				
				# Generate images
				generated_images = generator(batch_train, training=False)
				
				# Prepare labels
				valid = tf.ones((per_replica_batch_size, 8, 64, 1), dtype=tf.float32)
				fake = tf.zeros((per_replica_batch_size, 8, 64, 1), dtype=tf.float32)
				
				# Train discriminator_1
				with tf.GradientTape() as disc1_tape:
					# Real images
					real_pred = discriminator_1([batch_target, batch_train], training=True)
					real_loss = tf.keras.losses.binary_crossentropy(valid, real_pred)
					
					# Fake images
					fake_pred = discriminator_1([generated_images, batch_train], training=True)
					fake_loss = tf.keras.losses.binary_crossentropy(fake, fake_pred)
					
					d1_loss = (tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)) / 2
				
				# Apply discriminator_1 gradients
				d1_grads = disc1_tape.gradient(d1_loss, discriminator_1.trainable_variables)
				# Filter out None gradients
				d1_grads_and_vars = [(grad, var) for grad, var in zip(d1_grads, discriminator_1.trainable_variables) if grad is not None]
				if d1_grads_and_vars:
					disc1_optimizer.apply_gradients(d1_grads_and_vars)
				
				# Train discriminator_2 (CRNN)
				with tf.GradientTape() as disc2_tape:
					# Forward pass through CRNN model
					d2_predictions = discriminator_2(x_train_rcnn, training=True)
					
					# Ensure y_train_rcnn has correct shape for CTC loss
					# y_train_rcnn should be (batch_size, max_text_length)
					if len(y_train_rcnn.shape) == 1:
						y_train_rcnn = tf.expand_dims(y_train_rcnn, axis=0)
					
					# Pad or truncate to ensure consistent dimensions
					max_len = tf.reduce_max(tf.math.count_nonzero(y_train_rcnn, axis=-1))
					y_train_rcnn = y_train_rcnn[:, :max_len]
					
					# Calculate CTC loss manually
					d2_loss = ctc_loss_lambda_func(y_train_rcnn, d2_predictions)
					d2_loss = tf.reduce_mean(d2_loss)
				
				d2_grads = disc2_tape.gradient(d2_loss, discriminator_2.trainable_variables)
				# Filter out None gradients
				d2_grads_and_vars = [(grad, var) for grad, var in zip(d2_grads, discriminator_2.trainable_variables) if grad is not None]
				if d2_grads_and_vars:
					disc2_optimizer.apply_gradients(d2_grads_and_vars)
				
				# Train generator via GAN
				with tf.GradientTape() as gen_tape:
					# Use explicit call method to avoid training parameter conflict
					gan_outputs = gan.call([batch_train], training=True)
					# Unpack outputs: [discriminator_1_out, generator_out, discriminator_2_out]
					_, generator_out, crnn_out = gan_outputs
					
					# Calculate losses manually for better control
					# Generator loss (BCE) - need to reshape valid to match generator_out shape
					# valid shape: (12, 8, 64, 1) -> generator_out shape: (12, 128, 1024, 1)
					valid_reshaped = tf.image.resize(valid, [128, 1024])  # Resize to match generator output
					gen_loss = tf.keras.losses.binary_crossentropy(valid_reshaped, generator_out)
					
					# CRNN loss - ensure proper dimensions
					if len(y_train_rcnn.shape) == 1:
						y_train_rcnn_padded = tf.expand_dims(y_train_rcnn, axis=0)
					else:
						y_train_rcnn_padded = y_train_rcnn
					
					# Pad or truncate to ensure consistent dimensions
					max_len = tf.reduce_max(tf.math.count_nonzero(y_train_rcnn_padded, axis=-1))
					y_train_rcnn_padded = y_train_rcnn_padded[:, :max_len]
					
					crnn_loss = ctc_loss_lambda_func(y_train_rcnn_padded, crnn_out)
					
					# Combined loss (matching original GAN loss weights [1,10,1])
					# Cast both losses to the same type to avoid type mismatch
					gen_loss_float = tf.cast(tf.reduce_mean(gen_loss), tf.float32)
					crnn_loss_float = tf.cast(crnn_loss, tf.float32)
					g_loss = gen_loss_float * 10 + crnn_loss_float * 1
				
				gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
				# Filter out None gradients
				gen_grads_and_vars = [(grad, var) for grad, var in zip(gen_grads, generator.trainable_variables) if grad is not None]
				if gen_grads_and_vars:
					gen_optimizer.apply_gradients(gen_grads_and_vars)
				
				return d1_loss, d2_loss, g_loss
			
			# Run distributed training
			per_replica_losses = strategy.run(train_step, args=(batch_data,))
			
			# Reduce losses across replicas
			d1_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
			d2_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
			g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
			
			return d1_loss, d2_loss, g_loss
	
	# Main training loop
	for e in range(ep_start, epochs + 1):
		print(f"\n🔄 Epoch {e}/{epochs}")
		
		# Create optimized dataset
		dataset = create_optimized_dataset(list_image_train, list_lines, 'train', strategy, batch_size)
		
		# Distribute dataset across GPUs
		distributed_dataset = strategy.experimental_distribute_dataset(dataset)
		
		# Track progress for this epoch
		print(f"📊 Dataset optimized: global_batch={batch_size}, per_replica={batch_size // strategy.num_replicas_in_sync}")
		print(f"🔧 Using {strategy.num_replicas_in_sync} GPUs with advanced pipeline optimizations")
		
		# Training loop with performance monitoring
		nb = 0
		epoch_start_time = time.time()
		
		for batch_data in distributed_dataset:
			batch_start_time = time.time()
			
			# Distributed training step
			d1_loss, d2_loss, g_loss = distributed_train_step(batch_data)
			
			nb += 1
			
			# Advanced memory management
			if nb % 100 == 0:  # Less frequent clearing for better performance
				gc.collect()
				print(f"🧹 Memory optimized at batch {nb}")
			
			# Progress reporting with performance metrics
			if nb % 10 == 0:
				batch_time = time.time() - batch_start_time
				samples_per_second = batch_size / batch_time
				
				print(f'⚡ Batch {nb} - D1: {d1_loss:.4f}, D2: {d2_loss:.4f}, G: {g_loss:.4f} '
					  f'| Speed: {samples_per_second:.1f} samples/sec | Time: {batch_time:.2f}s')
		
		epoch_time = time.time() - epoch_start_time
		print(f'✅ Epoch {e} completed in {epoch_time:.1f}s')
		
		# Save models every 25 epochs
		if e % 25 == 0:
			save_epoch = e
			print(f"💾 Saving models at epoch {save_epoch}")
			try:
				generator.save_weights(f'checkpoints/generator_epoch_{save_epoch}.weights.h5')
				discriminator_1.save_weights(f'checkpoints/discriminator1_epoch_{save_epoch}.weights.h5')
				discriminator_2.save_weights(f'checkpoints/discriminator2_epoch_{save_epoch}.weights.h5')
				gan.save_weights(f'checkpoints/gan_epoch_{save_epoch}.weights.h5')
			except Exception as e_save:
				print(f"⚠️ Error saving models: {e_save}")
			
		# Evaluate every few epochs
		if e <= 5 or e % 4 == 0:
			evaluate(e, generator, discriminator_1, discriminator_2, gan)
	
	return generator, discriminator_1, discriminator_2, gan

def save(gan, generator, discriminator_1,discriminator_2,epoch):

	gan.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/gan.weights.h5")	
	
	discriminator_1.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/discriminator.weights.h5")
	discriminator_2.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn.weights.h5")
	#discriminator_3.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn_progressive.weights.h5")
	generator.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator.weights.h5")

def load(epoch):
	generator = unet()
	generator = generator.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")
	discriminator_1 = build_discriminator_1()
	discriminator_1.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/discriminator_weights.h5")
	 
	discriminator_2 = build_discriminator_2()
	discriminator_2.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn_weights.h5")
	 
	adam = get_optimizer()
	gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	
	#gan = gan.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/gan_weights.h5")
	return gan, generator, discriminator_1,discriminator_2
def evaluate(epoch, generator, discriminator_1,discriminator_2,gan):
	
	list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')
	#res = list_image_valid[-2:] 
	res = list_image_valid
	list_lines = read_file(rootPath + 'Sets/lines.txt')
	count_image=0
	for im_base in res:
		# Find the full filename with extension in the directory
		search_pattern = os.path.join('datasets/nan_distorted/validation', im_base + '.*')
		found_files = glob(search_pattern)
		
		if not found_files:
			# print(f"Warning: No image file found for base name {im_base} in validation set. Skipping.")
			continue
		
		im_full_name = os.path.basename(found_files[0])

		if count_image >=0:
			space = np.zeros((128,1024))
			deg_image, gt_image = readGrayPair(im_full_name, split='validation')

			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction.png", prediction, cmap='gray')
			plt.imsave("deg_image.png", np.squeeze(deg_image), cmap='gray')
			plt.imsave("gt_image.png", np.squeeze(gt_image), cmap='gray')
			plt.imsave("space.png", space, cmap='gray')
			im1=cv2.imread("prediction.png")
			im2=cv2.imread("deg_image.png")
			im3=cv2.imread("gt_image.png")
			im4=cv2.imread("space.png")
			show = vconcat_resize([im2, im4, im1, im4, im3])
		
			if not os.path.exists(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch)):
				os.makedirs(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch))
				os.makedirs(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights")
			cv2.imwrite(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + '/'+  im_full_name + ".png", show)
	save(gan, generator, discriminator_1,discriminator_2,epoch)
def train_GAN_crnn(nepochs,batch_size=12):
	global strategy
	# Validasi batch size terhadap jumlah replica
	if batch_size % strategy.num_replicas_in_sync != 0:
		adjusted = (batch_size // strategy.num_replicas_in_sync) * strategy.num_replicas_in_sync
		print(f"⚠️  batch_size {batch_size} tidak habis dibagi {strategy.num_replicas_in_sync} replica. Disesuaikan menjadi {adjusted}.")
		batch_size = adjusted if adjusted > 0 else strategy.num_replicas_in_sync

	print(f'🔧 Membangun model dalam strategy.scope() (replicas={strategy.num_replicas_in_sync}) ...')
	with strategy.scope():
		print('generator creation..............')
		generator = unet()
		print('discriminator 1 creation..............')
		discriminator_1 = build_discriminator_1()
		print('discriminator 2 (CRNN) creation..............')
		discriminator_2 = build_discriminator_2()
		# discriminator_3 saat ini tidak dipakai dalam training loop
		# print('discriminator 3 creation..............')
		# discriminator_3 = build_discriminator_3()
		adam = get_optimizer()
		gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	# Lanjutkan ke proses training
	generator, discriminator_1, discriminator_2, gan = train_gan(generator, discriminator_1, discriminator_2, gan, ep_start=0, epochs=nepochs, batch_size=batch_size)
def resume_train_GAN_crnn(nepochs,epo,batch_size=12):
	global strategy
	if batch_size % strategy.num_replicas_in_sync != 0:
		adjusted = (batch_size // strategy.num_replicas_in_sync) * strategy.num_replicas_in_sync
		print(f"⚠️  batch_size {batch_size} tidak habis dibagi {strategy.num_replicas_in_sync} replica. Disesuaikan menjadi {adjusted}.")
		batch_size = adjusted if adjusted > 0 else strategy.num_replicas_in_sync

	with strategy.scope():
		generator = unet()
		generator.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/generator_weights.h5")
		discriminator_1 = build_discriminator_1()
		discriminator_1.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/discriminator_weights.h5")
		discriminator_2 = build_discriminator_2()
		discriminator_2.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/rcnn_weights.h5")
		adam = get_optimizer()
		gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	generator, discriminator_1, discriminator_2, gan = train_gan(generator, discriminator_1, discriminator_2, gan, ep_start=epo, epochs=nepochs, batch_size=batch_size)

def loadCRNNModel(epoch,mode_crnn='no_progressive', batch_size=12):
	from data.generator import DataGenerator
	input_size = (1024, 128, 1)
	dtgen = DataGenerator(source=source_path,
						batch_size=batch_size,  # Use global batch_size
						charset=charset_base,
						max_text_length=max_text_length)


	from network.model import HTRModel

	# create and compile HTRModel
	model = HTRModel(architecture=arch,
					 input_size=input_size,
					 vocab_size=dtgen.tokenizer.vocab_size,
					 beam_width=10,
					 stop_tolerance=20,
					 reduce_tolerance=15)

	model.compile(learning_rate=0.001)
	model.summary(output_path, "summary.txt")

	# get default callbacks and load checkpoint weights file (HDF5) if exists
	if mode_crnn=='progressive':
		model.load_checkpoint(target='handwritten-text-recognition/ResultGanS3_iam_OP/epoch128/weights/rcnn_weights.h5')
	else:
		model.load_checkpoint(target='handwritten-text-recognition/output-IAM-GT/iam/flor/checkpoint_weights.hdf5')
	return dtgen,model
def ocr_crnn(filename,dtgen,model):
	text = ''
	input_size = (1024, 128, 1)

	im=pp.preprocess(filename,input_size)
	x_test = []
	x_test.append(im)
	x_test=pp.normalization(x_test)

	# predict() function will return the predicts with the probabilities
	predicts, _ = model.predict(x=x_test,
								use_multiprocessing=False,
								ctc_decode=True,
								verbose=0)

	# decode to string
	predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
	text=predicts[0]
	s=text.split()
	# s=s[::-1] # Removed for English
	reco=' '.join(s)
	reco=reco.strip()
	print(reco)
	return reco
def predict_gan(epoch, generator,list_image_valid,set):
	
	count_image=0
	for im in list_image_valid:
		if count_image >=0:

			#deg_image, gt_image = readGrayPairPad(im)
			original_path_image_gt=os.path.join(DatabasePath, set, 'images', im)
			claen_image=cv2.imread(original_path_image_gt)
			noisy_image_path=os.path.join('datasets/nan_distorted/', set, im)
			noisy_image=cv2.imread(noisy_image_path)
			
			#height, width,c = noisy_image.shape
			#############resize the height of noisy image to 32
			############add padding

			#noisy_image=addpad_image(noisy_image)
			height, width,c = noisy_image.shape
			#cv2.imwrite('out_padded.png',noisy_image)
			##############end padding
			
			original_image = Image.open(noisy_image_path) 
			original_image = original_image.resize((1024,128), Image.Resampling.LANCZOS)
			
			grey_image = original_image.convert('L')
			grey_image.save("deg_image3.png")
			deg_image = plt.imread("deg_image3.png")
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction3.png", prediction, cmap='gray')
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch)):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch))
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/prediction")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/prediction_reduced")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/visualize")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Truth")
			################"resize predicted image to original size
			cv2.imwrite(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/Truth/'+  im + ".png",claen_image)
			original_image = Image.open('prediction3.png') 
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/prediction_reduced/'+  im + ".png")
			########################""resizingggggggggg	
			original_image = original_image.resize((width,height), Image.Resampling.LANCZOS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/prediction/'+  im + ".png")
			# ######################space image
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted"):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted")
			
			original_image = Image.open(noisy_image_path) 
			original_image = original_image.resize((1024,128), Image.Resampling.LANCZOS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted/" + im + ".png")
			
		count_image=count_image+1
def predict_gan_hard(epoch, generator,list_image_valid,set):
	
	
	scenario='S_nan_OP'

	count_image=0
	for im in list_image_valid:
		if count_image >=0:

			#deg_image, gt_image = readGrayPairPad(im)
			original_path_image_gt=os.path.join(DatabasePath, set, 'images', im)
			claen_image=cv2.imread(original_path_image_gt)
			noisy_image_path=os.path.join('datasets/nan_distorted/', set, im)
			noisy_image=cv2.imread(noisy_image_path)
			
			#height, width,c = noisy_image.shape
			#############resize the height of noisy image to 32
			############add padding

			#noisy_image=addpad_image(noisy_image)
			height, width,c = noisy_image.shape
			#cv2.imwrite('out_padded.png',noisy_image)
			##############end padding
			
			original_image = Image.open(noisy_image_path) 
			original_image = original_image.resize((1024,128), Image.Resampling.LANCZOS)

			grey_image = original_image.convert('L')
			grey_image.save("deg_image3x.png")
			deg_image = plt.imread("deg_image3x.png")
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction3x.png", prediction, cmap='gray')
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch)):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch))
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/prediction")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/prediction_reduced")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/visualize")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/Truth")
			################"resize predicted image to original size
			#cv2.imwrite(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/Truth/'+  im + ".png",claen_image)
			original_image = Image.open('prediction3x.png') 
			#original_image.save(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/prediction_reduced/'+  im + ".png")
			########################""resizingggggggggg	
			original_image = original_image.resize((width,height), Image.Resampling.LANCZOS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/prediction/'+  im + ".png")
			######################space image
			
		count_image=count_image+1
def addpad_image(img):

	# convert each image of shape (32, 128, 1)
	w, h,c = img.shape
	#print(h)
	white =   [255,255,255]
	
	w_ad=1024-h
	if h < 1024:

		return cv2.copyMakeBorder(img,0,0,w_ad,0,cv2.BORDER_CONSTANT,value=white)
	else:
		return img	

	
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def get_psnr_iam():
	DatabasePathGT = 'handwritten-text-recognition/ResultsSauvegarde/ResultGanS2_W0p5/set_test_epoch_92/Truth/'

	count_image = 1
	qo = 0
	recap = 0
	list_image= read_file(rootPath + 'Sets/list_test_iam.txt')
	for im in list_image:
		original_path_image_gt = DatabasePathGT + '/' + im + '.png'
		original_image = Image.open(original_path_image_gt)
		original_image = original_image.resize((1024,128), Image.Resampling.LANCZOS)
		grey_image = original_image.convert('1')
		grey_image.save("gt.png")
		gt = plt.imread("gt.png")

		enhanced_image_path = 'handwritten-text-recognition/ResultGanS_iam_OP/hard3_set_test_epoch_112/prediction/'+  im + ".png"
		im2 = Image.open(enhanced_image_path)
		im2 = im2.resize((1024,128), Image.Resampling.LANCZOS)
		im2 = im2.convert('1')
		im2.save('im2.png')
		predicted = plt.imread('im2.png')

		psnrv = psnr(predicted, gt)
		print(psnrv)
		recap = recap + psnrv
		qo += 1
	av = recap / qo
	print('average psnr: ')
	print(av)
if __name__ == '__main__':
	replicas = strategy.num_replicas_in_sync
	print(f"🚀 Starting FULL OPTIMIZATION training with {replicas} GPU(s)")
	target_global_batch = batch_size  # Use the global batch_size variable
	if replicas == 1:
		print("ℹ️  Hanya 1 GPU terdeteksi (cek CUDA_VISIBLE_DEVICES jika mengharapkan multi-GPU).")
	else:
		per_rep = target_global_batch // replicas
		print(f"📊 Configuration: {replicas} GPUs × {per_rep} = {per_rep * replicas} global batch size")
	print("⚡ Mixed precision + XLA enabled")
	train_GAN_crnn(150, target_global_batch)


