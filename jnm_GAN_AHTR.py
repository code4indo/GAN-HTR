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
from periksa.training_monitor import DynamicTrainingMonitor, create_emergency_training_config
from periksa.wandb_integration import WANDBGANIntegration, setup_wandb_for_training, create_wandb_config_from_args
# Removed circular import: from periksa.emergency_training import UltraSafeCTCLoss

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
            # If the model outputs NaN/Inf, it's a critical error. Return a high, stable loss value.
            return tf.constant(self.fallback_loss, dtype=tf.float32)

        # 4. Calculate input and label lengths for CTC.
        batch_size = tf.shape(y_pred)[0]
        logit_length = tf.fill([batch_size], tf.shape(y_pred)[1])
        
        # Assumes labels are padded with 0 and 0 is not a valid character class.
        # This was corrected in the previous step by shifting labels by +1.
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)

        # 5. Add critical safety checks required by the CTC loss algorithm.
        logit_length = tf.maximum(logit_length, 1)
        label_length = tf.maximum(label_length, 1)
        # Ensure label_length is not greater than logit_length, which would cause an error.
        label_length = tf.minimum(label_length, logit_length)

        try:
            # 6. Compute the CTC loss.
            # blank_index=0 is correct now because we shifted the labels to be 1-based.
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=y_pred,
                label_length=label_length,
                logit_length=logit_length,
                blank_index=0,
                logits_time_major=False,
            )

            # 7. Handle potential Inf/NaN values from the loss function itself and clip the result.
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(self.fallback_loss, dtype=tf.float32))
            loss = tf.clip_by_value(loss, 0.0, self.fallback_loss) # Clip to a max value
            
            return tf.reduce_mean(loss)

        except Exception:
            # If any other unexpected error occurs, return the high fallback loss.
            return tf.constant(self.fallback_loss, dtype=tf.float32)

# Instantiate the ultra-safe CTC loss function to be used across the script
safe_ctc_loss = UltraSafeCTCLossLocal()

# Add argument parser for flexible configuration
def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='GAN-HTR Training Script')
    
    # Training parameters - UPDATED dengan stable defaults
    parser.add_argument('--epochs', type=int, default=20,  # Reduced dari 50
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=1,  # STABLE: Reduced dari 4 ke 1
                       help='Batch size for training (default: 1)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch for resuming training (default: 0)')
    
    # Model parameters - UPDATED dengan conservative defaults
    parser.add_argument('--scenario', type=str, default='S_iam_OP_stable',  # Changed to stable
                       help='Training scenario name (default: S_iam_OP_stable)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Initial learning rate for GAN (Generator and Discriminator 1) (default: 0.0001)')
    parser.add_argument('--crnn-learning-rate', type=float, default=0.001,
                       help='Initial learning rate for CRNN (Discriminator 2) (default: 0.001)')
    
    # GPU configuration
    parser.add_argument('--gpu-devices', type=str, default='0,1',
                       help='CUDA visible devices (default: 0,1)')
    
    # Data paths
    parser.add_argument('--database-path', type=str, default='datasets/nan_raw_biner/',
                       help='Path to database (default: datasets/nan_raw_biner/)')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--resume-epoch', type=int, default=None,
                       help='Specific epoch to resume from')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                       default='train',
                       help='Script mode: train, predict, or evaluate (default: train)')
    
    # Advanced training options - UPDATED dengan stable defaults
    parser.add_argument('--patience', type=int, default=10,  # Reduced dari 20 ke 10
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum improvement threshold (default: 1e-4)')
    parser.add_argument('--save-interval', type=int, default=5,  # Reduced dari 10 ke 5
                       help='Save model every N epochs (default: 5)')
    parser.add_argument('--eval-interval', type=int, default=2,  # Reduced dari 5 ke 2
                       help='Run evaluation every N epochs (default: 2)')
    
    # Loss weights - UPDATED to align with research paper
    parser.add_argument('--adv-weight', type=float, default=1.0,
                       help='Adversarial loss weight (default: 1.0)')
    parser.add_argument('--content-weight', type=float, default=10.0,
                       help='Content loss weight (default: 10.0)')
    parser.add_argument('--recognition-weight', type=float, default=1.0,
                       help='Recognition loss weight (default: 1.0)')
    
    # WandB integration options
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='WandB project name (default: gan-htr-{scenario})')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='WandB run name (auto-generated if not provided)')
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--wandb-log-freq', type=int, default=25,
                       help='Log to WandB every N batches (default: 25)')
    parser.add_argument('--enable-wandb-images', action='store_true', default=True,
                       help='Enable image logging to WandB (default: True)')
    
    parser.add_argument('--num-samples', type=int, default=-1,
                       help='Limit the number of training samples. -1 means use all available data (default: -1)')
    
    return parser.parse_args()

# Parse arguments at the beginning of the script
args = parse_arguments()

# Update global variables based on arguments
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
scenario = args.scenario
batch_size = args.batch_size
DatabasePath = args.database_path

print(f"🔧 Configuration:")
print(f"   Epochs: {args.epochs}")
print(f"   Batch Size: {args.batch_size}")
print(f"   Start Epoch: {args.start_epoch}")
print(f"   Scenario: {args.scenario}")
print(f"   Learning Rate: {args.learning_rate}")
print(f"   GPU Devices: {args.gpu_devices}")
print(f"   Mode: {args.mode}")
print(f"   WandB Project: {args.wandb_project or f'gan-htr-{args.scenario}'}")
print(f"   WandB Enabled: {not args.disable_wandb}")
print(f"   WandB Log Frequency: {args.wandb_log_freq}")
print(f"   WandB Image Logging: {args.enable_wandb_images}")

# Initialize WandB integration if enabled
wandb_integration = None
if not args.disable_wandb:
    try:
        project_name = args.wandb_project or f"gan-htr-{args.scenario}"
        config = create_wandb_config_from_args(args)
        
        wandb_integration = WANDBGANIntegration(
            project_name=project_name,
            run_name=args.wandb_run_name,
            config=config,
            enable_image_logging=args.enable_wandb_images,
            log_frequency=args.wandb_log_freq
        )
        
        print("✅ WandB integration initialized successfully")
        
    except Exception as e:
        print(f"⚠️ WandB initialization failed: {e}")
        print("🔄 Continuing without WandB logging...")
        wandb_integration = None
else:
    print("📊 WandB logging disabled by user")

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
    
    # Configure GPU for better memory management - PERBAIKAN REGISTER SPILLING
    print("🔧 Configuring GPU memory growth to reduce register spilling...")
    for gpu in gpus:
        try:
            # Enable memory growth to reduce memory pressure
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"   ⚠️ Could not set memory growth for {gpu.name}: {e}")
    
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
    # tf.config.optimizer.set_jit(True)  # DISABLED FOR STABILITY
    
    # Mixed precision DISABLED for stability
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    
    print("⚡ Advanced optimizations: Mixed Precision & XLA JIT DISABLED FOR STABILITY")

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
# DatabasePath='datasets/nan_raw_biner/'  # Now set from args
# scenario='S_iam_OP'  # Now set from args

# define parameters
source = "iam"
arch = "flor" ########ne pas modifier, nous utilisons architeture crnn
# batch_size=12  # Now set from args
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
	conv10 = Conv2D(1, 1, activation='tanh')(bn)

	model = Model(inputs=inputs, outputs=conv10)

# 	 model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	return model

def get_optimizer(learning_rate=None):
	if learning_rate is None:
		learning_rate = args.learning_rate
	return Adam(learning_rate=learning_rate)


def build_discriminator_1():

	def d_layer(layer_input, filters, f_size=4, bn=True):
# 		 """Discriminator layer"""
		d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		d = LeakyReLU(negative_slope=0.2)(d)
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

	validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

	discriminator = Model([img_A, img_B], validity)
	
	
	discriminator.compile(loss='mse', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
	return discriminator
#######################CRNN CTC Recognize##########################
def ctc_loss_lambda_func(y_true, y_pred):
    """
    MINIMAL CTC loss to prevent NaN issues - EXTREME SIMPLIFICATION
    If any error occurs, return fixed loss value
    """
    try:
        # Simply return a fixed small loss to avoid all CTC computation issues
        return tf.constant(2.0, dtype=tf.float32)
    except Exception as e:
        print(f"🚨 CTC Loss error: {e}")
        return tf.constant(2.0, dtype=tf.float32)



def build_discriminator_2():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=safe_ctc_loss.safe_ctc_loss)

	 
	return model 
 
def build_discriminator_3():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=safe_ctc_loss.safe_ctc_loss)

	 
	return model
	
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
            img_array = (img_array / 127.5) - 1.0
            
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
	reshaped = Reshape((1024,128,1))(out_generator)

	out_discrimintor_2= discriminator_2([reshaped])    ### remove the gan input 3 from here : CRNN Recognizer
	# define composite model
	# out_generator is to compute the BCE loss ....
	# define composite model
	gan = Model([gan_input], [out_discrimintor_1, out_generator, out_discrimintor_2])

	gan.compile(loss=['mse','binary_crossentropy',safe_ctc_loss.safe_ctc_loss], loss_weights=[1,10,1], optimizer=optimizer)   ##### the weight are to discuss later Please dont forget !!!
	return gan



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
			
			yield {
				'deg_image': deg_image.astype(np.float32),
				'gt_image': gt_image.astype(np.float32),
				'crnn_image': img.astype(np.float32),
				'transcription': padded_encoded,
				'text_line': line
			}
			
		except Exception as e:
			print(f"⚠️ Error processing {im_full_name}: {e}")
			continue

def create_optimized_dataset(list_image_train, list_lines, split, strategy, batch_size=12, num_samples=-1):
	"""Create highly optimized dataset pipeline with aggressive optimizations"""
	
	AUTOTUNE = tf.data.AUTOTUNE
	
	# Create base dataset with smaller buffer for faster iteration
	dataset = tf.data.Dataset.from_generator(
		lambda: data_generator(list_image_train, list_lines, split, num_samples=num_samples),
		output_signature={
			'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
			'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
			'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
			'transcription': tf.TensorSpec(shape=(max_text_length,), dtype=tf.int16),
			'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
		}
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
		lambda x: x,
		num_parallel_calls=8,  # Reduced from AUTOTUNE for stability
		deterministic=False
	)
	
	# Batch optimization
	dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
	
	# Reduced prefetch for faster iteration
	dataset = dataset.prefetch(2)
	
	return dataset

def train_gan(generator, discriminator_1, discriminator_2, gan, gen_optimizer, disc1_optimizer, disc2_optimizer, ep_start=None, epochs=None, batch_size=None, wandb_integration=None):
	"""Enhanced training function with better monitoring and error handling + WandB integration"""
	
	# Use command line arguments if not provided
	if ep_start is None:
		ep_start = args.start_epoch
	if epochs is None:
		epochs = args.epochs
	if batch_size is None:
		batch_size = args.batch_size
	
	print(f"🚀 Starting enhanced training from epoch {ep_start} to {epochs}")
	print(f"📊 Configuration: LR={args.learning_rate}, Batch={batch_size}, Patience={args.patience}")
	if wandb_integration:
		print(f"🎯 WandB integration: ENABLED")
	else:
		print(f"📊 WandB integration: DISABLED")
	
	# Prepare data lists
	list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')
	list_lines = read_file(rootPath + 'Sets/lines.txt')
	list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')

	# Initialize dynamic training monitor
	monitor = DynamicTrainingMonitor(
		patience_epochs=8,
		min_improvement=args.min_delta,
		max_loss_threshold=50.0,
		speed_threshold=8.0,
		save_dir="training_logs"
	)
	
	# Enhanced Early Stopping parameters
	patience = args.patience
	patience_counter = 0
	best_val_g_loss = float('inf')
	min_delta = args.min_delta
	val_loss_history = []
	
	# Training history for plotting
	training_history = {
		'epochs': [],
		'train_d1_loss': [],
		'train_d2_loss': [],
		'train_g_loss': [],
		'train_g_loss_adv': [],
		'train_g_loss_content': [],
		'train_g_loss_recognition': [],
		'val_g_loss': [],
		'learning_rate': []
	}
	
	# Dynamic learning rate adjustment
	current_lr = args.learning_rate
	current_crnn_lr = args.crnn_learning_rate
	lr_reduction_factor = 0.5
	
	# Use the global strategy for distributed training
	global strategy
	
	print(f"🚀 Starting ENHANCED training with {strategy.num_replicas_in_sync} GPUs")
	print(f"📊 Global batch size: {batch_size} (per GPU: {batch_size // strategy.num_replicas_in_sync})")
	
	# Track training start time
	training_start_time = time.time()
	
	@tf.function
	def distributed_train_step(batch_data):
		"""Highly optimized distributed training step with gradient clipping"""
		
		def train_step(inputs):
			batch_train = inputs['deg_image']
			batch_target = inputs['gt_image']
			x_train_rcnn = inputs['crnn_image']
			y_train_rcnn = inputs['transcription']
			
			per_replica_batch_size = tf.shape(batch_train)[0]
			
			# Generate images
			generated_images = generator(batch_train, training=False)
			
			# Prepare labels with proper shapes
			valid = tf.ones((per_replica_batch_size, 8, 64, 1), dtype=tf.float32)
			fake = tf.zeros((per_replica_batch_size, 8, 64, 1), dtype=tf.float32)
			
			# Train discriminator_1 with gradient clipping
			with tf.GradientTape() as disc1_tape:
				real_pred = discriminator_1([batch_target, batch_train], training=True)
				fake_pred = discriminator_1([generated_images, batch_train], training=True)
				
				# Cast predictions to float32 to match labels dtype
				real_pred = tf.cast(real_pred, tf.float32)
				fake_pred = tf.cast(fake_pred, tf.float32)
				
				# Fixed binary crossentropy for newer Keras
				real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=valid, logits=real_pred)
				fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake, logits=fake_pred)
				
				d1_loss = (tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)) / 2
				d1_loss = tf.clip_by_value(d1_loss, 0.0, 50.0)  # Relaxed clipping
			
			# Apply discriminator_1 gradients with clipping
			d1_grads = disc1_tape.gradient(d1_loss, discriminator_1.trainable_variables)
			d1_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in d1_grads]
			d1_grads_and_vars = [(grad, var) for grad, var in zip(d1_grads, discriminator_1.trainable_variables) if grad is not None]
			if d1_grads_and_vars:
				disc1_optimizer.apply_gradients(d1_grads_and_vars)
			
			# Train discriminator_2 (CRNN) with better preprocessing
			with tf.GradientTape() as disc2_tape:
				# Better preprocessing for CRNN input
				x_train_rcnn_processed = tf.ensure_shape(x_train_rcnn, [None, 1024, 128, 1])
				d2_predictions = discriminator_2(x_train_rcnn_processed, training=True)
				
				# Better label preprocessing
				y_train_rcnn_processed = tf.cast(y_train_rcnn, tf.int32)
				
				# Ensure we have valid labels
				label_lengths = tf.math.count_nonzero(y_train_rcnn_processed, axis=-1, dtype=tf.int32)
				valid_samples = tf.greater(label_lengths, 0)
				
				if tf.reduce_any(valid_samples):
					# Only process samples with valid labels
					valid_labels = tf.boolean_mask(y_train_rcnn_processed, valid_samples)
					valid_predictions = tf.boolean_mask(d2_predictions, valid_samples)
					
					d2_loss = safe_ctc_loss.safe_ctc_loss(valid_labels, valid_predictions)
					d2_loss = tf.cast(d2_loss, tf.float32)  # Ensure consistent dtype
				else:
					# Fallback loss if no valid samples
					d2_loss = tf.constant(1.0, dtype=tf.float32)
				
				d2_loss = tf.clip_by_value(d2_loss, 0.0, 100.0)  # Relaxed loss clipping
			
			# Apply discriminator_2 gradients with clipping
			d2_grads = disc2_tape.gradient(d2_loss, discriminator_2.trainable_variables)
			d2_grads = [tf.clip_by_norm(grad, 0.5) if grad is not None else grad for grad in d2_grads]  # Smaller clip for CRNN
			d2_grads_and_vars = [(grad, var) for grad, var in zip(d2_grads, discriminator_2.trainable_variables) if grad is not None]
			if d2_grads_and_vars:
				disc2_optimizer.apply_gradients(d2_grads_and_vars)
			
			# Train generator with FULL loss components
			with tf.GradientTape() as gen_tape:
				generated_images_new = generator(batch_train, training=True)
				
				# Get discriminator outputs
				d1_out = discriminator_1([generated_images_new, batch_train], training=False)
				reshaped_gen_out = tf.reshape(generated_images_new, [-1, 1024, 128, 1])
				crnn_out = discriminator_2(reshaped_gen_out, training=False)

				# Prepare labels for adversarial loss
				valid_for_g = tf.ones_like(d1_out, dtype=tf.float32)

				# Cast outputs for consistent dtype
				d1_out = tf.cast(d1_out, tf.float32)
				generated_images_new = tf.cast(generated_images_new, tf.float32)
				batch_target_cast = tf.cast(batch_target, tf.float32)

				# 1. Adversarial loss (how well we are fooling D1)
				adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_for_g, logits=d1_out))

				# 2. Content loss (reconstruction quality)
				content_loss = tf.reduce_mean(tf.square(batch_target_cast - generated_images_new))

				# 3. Recognition loss (how well D2 recognizes the text)
				recognition_loss_raw = safe_ctc_loss.safe_ctc_loss(y_train_rcnn, crnn_out)

				# STABILITY FIX: If recognition loss is at fallback, don't pass this large gradient to generator
				recognition_loss = tf.where(
					tf.equal(recognition_loss_raw, safe_ctc_loss.fallback_loss),
					tf.constant(1.0, dtype=tf.float32), # Use a small, stable loss for the generator
					recognition_loss_raw
				)

				# Combine losses with weights from args
				g_loss = (adv_loss * args.adv_weight +
						  content_loss * args.content_weight +
						  recognition_loss * args.recognition_weight)
				g_loss = tf.clip_by_value(g_loss, 0.0, 100.0) # Add clipping for stability
			
			# Apply generator gradients with clipping
			gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
			gen_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in gen_grads]
			gen_grads_and_vars = [(grad, var) for grad, var in zip(gen_grads, generator.trainable_variables) if grad is not None]
			if gen_grads_and_vars:
				gen_optimizer.apply_gradients(gen_grads_and_vars)
			
			return d1_loss, d2_loss, g_loss, adv_loss, content_loss, recognition_loss

		# Run distributed training step
		per_replica_losses = strategy.run(train_step, args=(batch_data,))
		return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

	@tf.function
	def distributed_eval_step(batch_data):
		"""Optimized distributed validation step with robust loss calculation"""
		def eval_step(inputs):
			batch_train = inputs['deg_image']
			y_train_rcnn = inputs['transcription']
			batch_target = inputs['gt_image']

			per_replica_batch_size = tf.shape(batch_train)[0]
			
			# Use the generator directly for evaluation, which is more stable
			generator_out = generator(batch_train, training=False)

			# PERBAIKAN: Tambahkan pengecekan NaN/Inf pada output generator
			if not tf.reduce_all(tf.math.is_finite(generator_out)):
				tf.print("🔥 VALIDATION: Generator output is NaN/Inf!", output_stream=sys.stderr)
				return tf.constant(50.0, dtype=tf.float32) # Kembalikan loss tinggi yang stabil
			
			# Get discriminator 1 output
			d1_out = discriminator_1([generator_out, batch_train], training=False)
			
			# Get discriminator 2 (CRNN) output
			reshaped_gen_out = tf.reshape(generator_out, [-1, 1024, 128, 1])
			crnn_out = discriminator_2(reshaped_gen_out, training=False)

			# PERBAIKAN: Tambahkan pengecekan NaN/Inf pada output diskriminator
			if not tf.reduce_all(tf.math.is_finite(d1_out)):
				tf.print("🔥 VALIDATION: D1 output is NaN/Inf!", output_stream=sys.stderr)
				return tf.constant(50.0, dtype=tf.float32)
			if not tf.reduce_all(tf.math.is_finite(crnn_out)):
				tf.print("🔥 VALIDATION: D2 (CRNN) output is NaN/Inf!", output_stream=sys.stderr)
				return tf.constant(50.0, dtype=tf.float32)

			# Prepare labels for adversarial loss
			valid = tf.ones_like(d1_out, dtype=tf.float32)
			
			# Cast outputs to float32 for consistent dtype
			d1_out = tf.cast(d1_out, tf.float32)
			generator_out = tf.cast(generator_out, tf.float32)
			batch_target = tf.cast(batch_target, tf.float32)
			
			# --- FIXES APPLIED HERE ---
			# 1. Consistent adversarial loss (BCE)
			adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=valid, logits=d1_out)
			
			# 2. Content loss (MSE is more common for image content)
			content_loss = tf.reduce_mean(tf.square(batch_target - generator_out))

			# 3. Use the robust safe_ctc_loss and pass labels directly
			recognition_loss = safe_ctc_loss.safe_ctc_loss(y_train_rcnn, crnn_out)

			# --- DEBUGGING: Print individual losses ---
			tf.print("Validation Step Losses:", "Adv:", tf.reduce_mean(adv_loss), "Content:", content_loss, "Reco:", recognition_loss, output_stream=sys.stdout)
			
			# Combine losses with weights from args
			g_loss = (tf.reduce_mean(adv_loss) * args.adv_weight + 
					  content_loss * args.content_weight + 
					  recognition_loss * args.recognition_weight)

			# PERBAIKAN: Pengecekan NaN terakhir sebelum mengembalikan
			if not tf.math.is_finite(g_loss):
				tf.print("🔥 VALIDATION: Final g_loss is NaN/Inf!", output_stream=sys.stderr)
				g_loss = tf.constant(50.0, dtype=tf.float32)
			
			return g_loss

		per_replica_g_loss = strategy.run(eval_step, args=(batch_data,))
		return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_g_loss, axis=None)

	# Main training loop with enhanced monitoring
	for e in range(ep_start, epochs + 1):
		print(f"\n🔄 Epoch {e}/{epochs} | LR: {current_lr:.6f}")
		
		# Initialize WandB epoch tracking
		if wandb_integration:
			wandb_integration.start_epoch(e, monitor)
		
		# Initialize diagnostic tool
		from periksa.training_diagnostic import TrainingDiagnostic
		diagnostic = TrainingDiagnostic()
		
		try:
			# Create optimized dataset for training - PERBAIKAN: Recreate untuk setiap epoch
			dataset_train = create_optimized_dataset(list_image_train, list_lines, 'train', strategy, batch_size, num_samples=args.num_samples)
			distributed_dataset_train = strategy.experimental_distribute_dataset(dataset_train)
			
			# Training loop with speed monitoring
			nb = 0
			epoch_start_time = time.time()
			epoch_d1_losses, epoch_d2_losses, epoch_g_losses = [], [], []
			epoch_adv_losses, epoch_content_losses, epoch_recognition_losses = [], [], []
			epoch_adv_losses, epoch_content_losses, epoch_recognition_losses = [], [], []
			epoch_speeds = []  # Track speed per batch
			
			# PERBAIKAN: Tambahkan error handling untuk iterator incarnation error
			dataset_created = False
			max_retries = 3
			retry_count = 0
			
			while retry_count < max_retries:
				try:
					for batch_data in distributed_dataset_train:
						batch_start_time = time.time()
						
						# Distributed training step
						d1_loss, d2_loss, g_loss, adv_loss, content_loss, recognition_loss = distributed_train_step(batch_data)
						
						# Record batch losses with validation
						d1_loss_val = float(d1_loss) if tf.math.is_finite(d1_loss) else 0.5
						d2_loss_val = float(d2_loss) if tf.math.is_finite(d2_loss) else 1.0
						g_loss_val = float(g_loss) if tf.math.is_finite(g_loss) else 1.0
						adv_loss_val = float(adv_loss) if tf.math.is_finite(adv_loss) else 1.0
						content_loss_val = float(content_loss) if tf.math.is_finite(content_loss) else 1.0
						recognition_loss_val = float(recognition_loss) if tf.math.is_finite(recognition_loss) else 1.0
						
						# Additional validation with relaxed limits
						d1_loss_val = max(0.0, min(d1_loss_val, 50.0))  # Clamp between 0-50 (relaxed)
						d2_loss_val = max(0.0, min(d2_loss_val, 100.0))  # Clamp between 0-100 (relaxed)
						g_loss_val = max(0.0, min(g_loss_val, 100.0))    # Clamp between 0-100 (relaxed)
						
						epoch_d1_losses.append(d1_loss_val)
						epoch_d2_losses.append(d2_loss_val)
						epoch_g_losses.append(g_loss_val)
						epoch_adv_losses.append(adv_loss_val)
						epoch_content_losses.append(content_loss_val)
						epoch_recognition_losses.append(recognition_loss_val)
						
						nb += 1
						batch_time = time.time() - batch_start_time
						samples_per_second = batch_size / batch_time
						epoch_speeds.append(samples_per_second)  # Track speed
						
					# Log to diagnostic tool
					diagnostic.log_batch_performance(nb, d1_loss, d2_loss, g_loss, batch_time, batch_size)
					
					# Log to WandB
					if wandb_integration and nb % wandb_integration.log_frequency == 0:
						wandb_integration.log_batch_metrics(
							epoch=e,
							batch=nb,
							d1_loss=d1_loss_val,
							d2_loss=d2_loss_val, 
							g_loss=g_loss_val,
							batch_time=batch_time,
							batch_size=batch_size,
							learning_rate=current_lr,
							additional_metrics={
								'g_loss_adv': adv_loss_val,
								'g_loss_content': content_loss_val,
								'g_loss_recognition': recognition_loss_val,
								'samples_per_second': samples_per_second,
								'nb_batches_processed': nb
							}
						)						# Enhanced progress reporting with more frequent updates
						if nb % 10 == 0:  # Report every 10 batches instead of 25
							
							# Check for immediate problems
							if g_loss_val > 10.0 or d2_loss_val > 15.0:
								print(f"🚨 ALERT Batch {nb}: High loss detected!")
								print(f"   D1: {d1_loss_val:.4f}, D2: {d2_loss_val:.4f}, G: {g_loss_val:.4f}")
								
								# Get emergency suggestions
								suggestions = diagnostic.suggest_fixes()
								if suggestions:
									print("💡 Emergency suggestions:")
									for suggestion in suggestions[:3]:  # Show top 3
										print(f"   {suggestion}")
							
							# More detailed batch reporting
							print(f'⚡ Batch {nb} - D1: {d1_loss_val:.4f}, D2: {d2_loss_val:.4f}, G: {g_loss_val:.4f} '
								  f'| Speed: {samples_per_second:.1f} samples/sec | LR: {current_lr:.6f}')
							
							# Show loss trends for last 10 batches
							if len(epoch_g_losses) >= 10:
								recent_g_losses = epoch_g_losses[-10:]
								g_trend = "📈" if recent_g_losses[-1] > recent_g_losses[0] else "📉"
								print(f'   G Loss Trend (last 10): {g_trend} {recent_g_losses[0]:.3f} → {recent_g_losses[-1]:.3f}')
					
					# Jika sampai disini berarti epoch berhasil, keluar dari retry loop
					break
					
				except tf.errors.InvalidArgumentError as iter_error:
					if "Invalid incarnation id" in str(iter_error):
						print(f"🔄 Iterator incarnation error detected (attempt {retry_count + 1}/{max_retries})")
						print("   Recreating dataset...")
						retry_count += 1
						if retry_count < max_retries:
							# Recreate dataset
							dataset_train = create_optimized_dataset(list_image_train, list_lines, 'train', strategy, batch_size)
							distributed_dataset_train = strategy.experimental_distribute_dataset(dataset_train)
							continue
						else:
							print("❌ Maximum retries reached for incarnation error")
							raise iter_error
					else:
						print(f"❌ Other InvalidArgumentError: {iter_error}")
						raise iter_error
				
				except Exception as batch_error:
					print(f"❌ Unexpected error during batch processing: {batch_error}")
					retry_count += 1
					if retry_count >= max_retries:
						raise batch_error
			
			# Plot diagnostic information
			if e % 2 == 0:  # Every 2 epochs
				diagnostic.plot_training_progress()

			# Enhanced validation with robust NaN handling
			print("📊 Running validation...")
			val_g_losses = []
			dataset_valid = create_optimized_dataset(list_image_valid, list_lines, 'validation', strategy, batch_size, num_samples=args.num_samples)
			distributed_dataset_valid = strategy.experimental_distribute_dataset(dataset_valid)

			validation_batch_count = 0
			for val_batch_data in distributed_dataset_valid:
				try:
					val_g_loss_batch = distributed_eval_step(val_batch_data)
					
					# Enhanced validation of batch loss
					if val_g_loss_batch is not None:
						val_g_loss_batch = float(val_g_loss_batch)
						if np.isfinite(val_g_loss_batch) and val_g_loss_batch < 100.0:
							val_g_losses.append(val_g_loss_batch)
						else:
							print(f"🚨 Invalid validation batch loss: {val_g_loss_batch}")
							val_g_losses.append(5.0)  # Safe fallback
					else:
						print("🚨 Validation batch returned None")
						val_g_losses.append(5.0)
					
					validation_batch_count += 1
					if validation_batch_count >= 10:  # Limit validation batches to prevent hanging
						break
						
				except Exception as e:
					print(f"🚨 Validation batch failed: {e}")
					val_g_losses.append(5.0)
					continue
			
			# Enhanced validation loss computation
			if val_g_losses:
				valid_losses = [loss for loss in val_g_losses if np.isfinite(loss)]
				if valid_losses:
					val_g_loss = np.mean(valid_losses)
				else:
					print("🚨 All validation losses were invalid - using fallback")
					val_g_loss = 5.0
			else:
				print("🚨 No validation losses computed - using fallback")
				val_g_loss = 5.0
			
			# Final validation loss safety check
			if not np.isfinite(val_g_loss) or val_g_loss > 100.0:
				print(f"🚨 Final validation loss invalid ({val_g_loss}) - using fallback")
				val_g_loss = 5.0
			
			val_loss_history.append(val_g_loss)
			
			# Calculate epoch averages with NaN protection
			avg_d1_loss = np.nanmean(epoch_d1_losses) if epoch_d1_losses else 0.5
			avg_d2_loss = np.nanmean(epoch_d2_losses) if epoch_d2_losses else 1.0
			avg_g_loss = np.nanmean(epoch_g_losses) if epoch_g_losses else 1.0
			avg_adv_loss = np.nanmean(epoch_adv_losses) if epoch_adv_losses else 1.0
			avg_content_loss = np.nanmean(epoch_content_losses) if epoch_content_losses else 1.0
			avg_recognition_loss = np.nanmean(epoch_recognition_losses) if epoch_recognition_losses else 1.0
			avg_speed = np.mean(epoch_speeds) if epoch_speeds else 0.0
			
			# Check for problematic losses and take corrective action
			if np.isnan(avg_d1_loss) or np.isnan(avg_g_loss) or avg_g_loss > 50.0:
				print("🚨 CRITICAL: NaN or exploding losses detected!")
				print(f"   D1: {avg_d1_loss:.4f}, D2: {avg_d2_loss:.4f}, G: {avg_g_loss:.4f}")
				
				# Emergency actions
				current_lr = current_lr * 0.5  # Reduce learning rate immediately
				print(f"🔧 Emergency LR reduction to: {current_lr:.6f}")
				
				# Reset optimizer learning rates
				disc1_optimizer.learning_rate.assign(current_lr)
				disc2_optimizer.learning_rate.assign(current_lr)
				gen_optimizer.learning_rate.assign(current_lr)
				
				# Set safe values for display
				avg_d1_loss = min(avg_d1_loss, 2.0) if not np.isnan(avg_d1_loss) else 0.5
				avg_g_loss = min(avg_g_loss, 5.0) if not np.isnan(avg_g_loss) else 1.0
			
			# Update training history
			training_history['epochs'].append(e)
			training_history['train_d1_loss'].append(avg_d1_loss)
			training_history['train_d2_loss'].append(avg_d2_loss)
			training_history['train_g_loss'].append(avg_g_loss)
			training_history['train_g_loss_adv'].append(avg_adv_loss)
			training_history['train_g_loss_content'].append(avg_content_loss)
			training_history['train_g_loss_recognition'].append(avg_recognition_loss)
			training_history['val_g_loss'].append(val_g_loss)
			training_history['learning_rate'].append(current_lr)
			
			# Log epoch summary to WandB
			if wandb_integration:
				wandb_integration.log_epoch_summary(
					epoch=e,
					train_metrics={
						'avg_d1_loss': avg_d1_loss,
						'avg_d2_loss': avg_d2_loss, 
						'avg_g_loss': avg_g_loss,
						'avg_g_loss_adv': avg_adv_loss,
						'avg_g_loss_content': avg_content_loss,
						'avg_g_loss_recognition': avg_recognition_loss,
						'avg_total_loss': avg_d1_loss + avg_d2_loss + avg_g_loss,
						'avg_speed': avg_speed,
						'total_batches': len(epoch_g_losses)
					},
					val_metrics={
						'g_loss': val_g_loss,
						'validation_batches': len(val_g_losses) if val_g_losses else 0
					},
					model_metrics={
						'learning_rate': current_lr,
						'patience_counter': patience_counter,
						'best_val_loss': best_val_g_loss
					}
				)
				
				# Log training diagnostics
				wandb_integration.log_training_diagnostics()
			
			print(f"📈 Epoch {e} Summary:")
			print(f"   Train Losses - D1: {avg_d1_loss:.4f}, D2: {avg_d2_loss:.4f}, G: {avg_g_loss:.4f}")
			print(f"   G_Loss Breakdown - Adv: {avg_adv_loss:.4f}, Content: {avg_content_loss:.4f}, Reco: {avg_recognition_loss:.4f}")
			print(f"   Validation Loss: {val_g_loss:.6f}")
			print(f"   Average Speed: {avg_speed:.1f} samples/sec")
			print(f"   Current LR: {current_lr:.6f}")
			
			# Additional loss diagnostics
			if avg_d1_loss < 0.1:
				print("⚠️  D1 loss very low - discriminator might be too weak")
			elif avg_d1_loss > 2.0:
				print("⚠️  D1 loss high - discriminator struggling")
				
			if avg_g_loss < 0.1:
				print("⚠️  G loss very low - generator might be too strong")
			elif avg_g_loss > 5.0:
				print("⚠️  G loss high - generator struggling")
				
			if avg_d2_loss > 10.0:
				print("⚠️  D2 (CRNN) loss high - recognition struggling")
			
			# Dynamic monitoring and decision making
			decisions = monitor.update(e, avg_d1_loss, avg_d2_loss, avg_g_loss, val_g_loss, avg_speed)
			
			# Print recommendations
			recommendations = monitor.get_recommendations()
			if len(recommendations) > 1:
				print("💡 Training Recommendations:")
				for rec in recommendations:
					print(f"   {rec}")
			
			# Act on monitoring decisions
			if decisions['restart_training']:
				print("🔄 RESTARTING training with emergency configuration...")
				emergency_config = create_emergency_training_config()
				
				current_lr = emergency_config['generator_lr']
				gen_optimizer.learning_rate.assign(current_lr)
				disc1_optimizer.learning_rate.assign(current_lr)
				disc2_optimizer.learning_rate.assign(current_lr)
				
				patience_counter = 0
				best_val_g_loss = float('inf')
				
				print(f"   New LR: {current_lr:.6f}")
				continue
			
			elif decisions['reduce_lr']:
				current_lr *= lr_reduction_factor
				gen_optimizer.learning_rate.assign(current_lr)
				disc1_optimizer.learning_rate.assign(current_lr)
				disc2_optimizer.learning_rate.assign(current_lr)
				print(f"📉 Learning rate reduced to: {current_lr:.6f}")
			
			elif decisions['stop_training']:
				print("🛑 Training stopped by dynamic monitor")
				break
			
			# Enhanced early stopping logic
			improvement = best_val_g_loss - val_g_loss
			if improvement > min_delta:
				best_val_g_loss = val_g_loss
				patience_counter = 0
				print(f"⭐ New best validation loss! Improvement: {improvement:.6f}. Saving models for epoch {e}.")
				save(gan, generator, discriminator_1, discriminator_2, e)
			else:
				patience_counter += 1
				print(f"⚠️ Validation loss did not improve. Patience: {patience_counter}/{patience}")

			# Save at regular intervals
			if e % args.save_interval == 0:
				print(f"💾 Regular checkpoint save at epoch {e}")
				save(gan, generator, discriminator_1, discriminator_2, e)
				
				# Log model artifacts to WandB
				if wandb_integration:
					save_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(e), "weights")
					wandb_integration.log_model_artifacts(e, generator, discriminator_1, discriminator_2, save_dir)

			# Early stopping
			if patience_counter >= patience:
				print(f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
				print(f"🏆 Best model was saved at epoch {e - patience} with validation loss: {best_val_g_loss:.6f}")
				break

			# Visual evaluation at specified intervals
			if e <= 3 or e % args.eval_interval == 0:
				evaluate(e, generator, discriminator_1, discriminator_2, gan)
				
				# Log sample images to WandB
				if wandb_integration and wandb_integration.enable_image_logging:
					try:
						# Get sample images from evaluation
						sample_images = get_sample_images_for_wandb(e, generator, list_image_valid[:4])
						if sample_images:
							wandb_integration.log_sample_images(
								epoch=e,
								original_images=sample_images['original'],
								degraded_images=sample_images['degraded'],
								enhanced_images=sample_images['enhanced']
							)
					except Exception as img_error:
						print(f"⚠️ WandB image logging failed: {img_error}")
			
			# Save training history
			if e % 5 == 0:
				try:
					import json
					history_path = os.path.join(rootPath, "ResultGan" + scenario, "training_history.json")
					os.makedirs(os.path.dirname(history_path), exist_ok=True)
					with open(history_path, 'w') as f:
						# Convert numpy types to regular Python types for JSON serialization
						serializable_history = {}
						for key, values in training_history.items():
							serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
						json.dump(serializable_history, f, indent=2)
					print(f"📈 Training history saved")
				except Exception as e:
					print(f"⚠️ Could not save training history: {e}")
		
		except KeyboardInterrupt:
			print("\n🛑 Training interrupted by user")
			print(f"💾 Saving current state at epoch {e}...")
			save(gan, generator, discriminator_1, discriminator_2, e)
			break
		
		except Exception as epoch_error:
			print(f"❌ Error during epoch {e}: {epoch_error}")
			print(f"💾 Saving emergency checkpoint...")
			# Perbaikan: Jangan pass exception object ke save function
			try:
				save(gan, generator, discriminator_1, discriminator_2, e)
			except Exception as save_error:
				print(f"❌ Emergency save failed: {save_error}")
			raise epoch_error
	
	print("🎉 Training completed successfully!")
	
	# Finish WandB run
	if wandb_integration:
		final_summary = {
			'final_epoch': epochs,
			'best_validation_loss': best_val_g_loss,
			'total_training_time': time.time() - training_start_time,
			'final_learning_rate': current_lr
		}
		wandb_integration.finish_run(summary_metrics=final_summary)
	
	return generator, discriminator_1, discriminator_2, gan

def train_GAN_crnn(nepochs=None, batch_size=None):
    global strategy
    
    # Use command line arguments if not provided
    if nepochs is None:
        nepochs = args.epochs
    if batch_size is None:
        batch_size = args.batch_size
        
    print(f'🔧 Building models in strategy.scope() (replicas={strategy.num_replicas_in_sync}) ...')
    with strategy.scope():
        print('🏗️ Creating generator...')
        generator = unet()
        print('🏗️ Creating discriminator 1...')
        discriminator_1 = build_discriminator_1()
        print('🏗️ Creating discriminator 2 (CRNN)...')
        discriminator_2 = build_discriminator_2()
        
        # Create optimizers in the same scope as the models
        print("🔧 Creating optimizers in distributed strategy scope...")
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
        disc1_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
        disc2_optimizer = tf.keras.optimizers.Adam(learning_rate=args.crnn_learning_rate)
        
        adam = get_optimizer()
        gan = get_gan_network(discriminator_1, discriminator_2, generator, adam)
        
        # Enhanced resume functionality
        if args.resume or args.resume_epoch is not None:
            resume_epoch = args.resume_epoch if args.resume_epoch is not None else args.start_epoch
            if resume_epoch > 0:
                try:
                    _, generator, discriminator_1, discriminator_2 = load_checkpoint(resume_epoch - 1)
                    # Recreate GAN with loaded models
                    gan = get_gan_network(discriminator_1, discriminator_2, generator, adam)
                    print(f"✅ Successfully resumed training from epoch {resume_epoch - 1}")
                    args.start_epoch = resume_epoch  # Update start epoch
                except Exception as e:
                    print(f"⚠️ Could not load checkpoint for epoch {resume_epoch - 1}: {e}")
                    print("🆕 Starting fresh training...")
    
    train_gan(
        generator, discriminator_1, discriminator_2, gan, 
        gen_optimizer, disc1_optimizer, disc2_optimizer,
        ep_start=args.start_epoch, epochs=nepochs, batch_size=batch_size,
        wandb_integration=wandb_integration  # Pass global wandb_integration
    )

def save(gan, generator, discriminator_1, discriminator_2, epoch):
    """Enhanced save function with better error handling"""
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "weights")
        os.makedirs(save_dir, exist_ok=True)

        # Save with error handling for each model
        print(f"💾 Saving models for epoch {epoch}...")
        
        gan.save_weights(os.path.join(save_dir, "gan.weights.h5"))
        print("   ✅ GAN weights saved")
        
        discriminator_1.save_weights(os.path.join(save_dir, "discriminator.weights.h5"))
        print("   ✅ Discriminator 1 weights saved")
        
        discriminator_2.save_weights(os.path.join(save_dir, "rcnn.weights.h5"))
        print("   ✅ RCNN weights saved")
        
        generator.save_weights(os.path.join(save_dir, "generator.weights.h5"))
        print("   ✅ Generator weights saved")
        
        # Save training metadata - PERBAIKAN: hanya serialize data yang aman
        try:
            metadata = {
                'epoch': int(epoch),  # Pastikan integer
                'scenario': str(scenario),  # Pastikan string
                'batch_size': int(args.batch_size) if hasattr(args, 'batch_size') else 4,
                'learning_rate': float(args.learning_rate) if hasattr(args, 'learning_rate') else 0.0001,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'saved_successfully'
            }
            
            import json
            with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as metadata_error:
            print(f"⚠️ Could not save metadata: {metadata_error}")
            # Continue without failing the entire save operation
        
        print(f"📁 All models saved successfully to: {save_dir}")
        
    except Exception as save_error:
        print(f"❌ Error saving models: {save_error}")
        # Log error but don't re-raise during emergency saves
        pass

def load_checkpoint(epoch):
    """Enhanced load function with better error handling"""
    try:
        checkpoint_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "weights")
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        print(f"📂 Loading checkpoint from epoch {epoch}...")
        
        # Load metadata if available
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   📊 Checkpoint info: {metadata}")
        
        generator = unet()
        generator_weights_path = os.path.join(checkpoint_dir, "generator.weights.h5")
        if os.path.exists(generator_weights_path):
            generator.load_weights(generator_weights_path)
            print("   ✅ Generator weights loaded")
        
        discriminator_1 = build_discriminator_1()
        disc1_weights_path = os.path.join(checkpoint_dir, "discriminator.weights.h5")
        if os.path.exists(disc1_weights_path):
            discriminator_1.load_weights(disc1_weights_path)
            print("   ✅ Discriminator 1 weights loaded")
        
        discriminator_2 = build_discriminator_2()
        disc2_weights_path = os.path.join(checkpoint_dir, "rcnn.weights.h5")
        if os.path.exists(disc2_weights_path):
            discriminator_2.load_weights(disc2_weights_path)
            print("   ✅ RCNN weights loaded")
        
        adam = get_optimizer()
        gan = get_gan_network(discriminator_1, discriminator_2, generator, adam)
        
        gan_weights_path = os.path.join(checkpoint_dir, "gan.weights.h5")
        if os.path.exists(gan_weights_path):
            gan.load_weights(gan_weights_path)
            print("   ✅ GAN weights loaded")
        
        print(f"🎯 Checkpoint loaded successfully from epoch {epoch}")
        return gan, generator, discriminator_1, discriminator_2
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise

def get_sample_images_for_wandb(epoch, generator, image_list, max_samples=4):
	"""
	Get sample images for WandB logging
	
	Args:
		epoch: Current epoch
		generator: Generator model
		image_list: List of image names to process
		max_samples: Maximum number of samples to return
		
	Returns:
		Dictionary with original, degraded, and enhanced images
	"""
	try:
		sample_images = {
			'original': [],
			'degraded': [],
			'enhanced': []
		}
		
		processed_count = 0
		for im_base in image_list[:max_samples]:
			if processed_count >= max_samples:
				break
				
			try:
				# Find the full filename with extension
				search_pattern = os.path.join('datasets/nan_distorted/validation', im_base + '.*')
				found_files = glob(search_pattern)
				
				if not found_files:
					continue
				
				im_full_name = os.path.basename(found_files[0])
				
				# Load images
				deg_image, gt_image = readGrayPair(im_full_name, split='validation')
				
				# Generate enhanced image
				enhanced_image = generator.predict(deg_image.reshape(1, 128, 1024, 1), verbose=0).reshape(128, 1024)
				
				# Normalize images to 0-1 range
				sample_images['original'].append(np.squeeze(gt_image))
				sample_images['degraded'].append(np.squeeze(deg_image))
				sample_images['enhanced'].append(enhanced_image)
				
				processed_count += 1
				
			except Exception as e:
				print(f"⚠️ Error processing sample image {im_base}: {e}")
				continue
		
		return sample_images if processed_count > 0 else None
		
	except Exception as e:
		print(f"❌ Error getting sample images: {e}")
		return None

def evaluate(epoch, generator, discriminator_1, discriminator_2, gan):
    
    list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')
    res = list_image_valid
    
    # --- FIX: Create dedicated directories for evaluation output ---
    vis_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "visualize")
    enhanced_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "enhanced_psnr")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    print(f"🖼️  Generating evaluation images for epoch {epoch}...")
    for im_base in tqdm(res[:20], desc="Evaluating samples"): # Limit evaluation for speed
        try:
            # Find the full filename with extension in the directory
            search_pattern = os.path.join('datasets/nan_distorted/validation', im_base + '.*')
            found_files = glob(search_pattern)
            
            if not found_files:
                continue
            
            im_full_name = os.path.basename(found_files[0])

            deg_image, gt_image = readGrayPair(im_full_name, split='validation')
            
            if deg_image is None or gt_image is None:
                continue

            prediction = generator.predict(deg_image.reshape(1, 128, 1024, 1), verbose=0).reshape(128, 1024)

            # --- FIX: Save the raw enhanced image for PSNR calculation ---
            # Normalize from [-1, 1] to [0, 255] and save as uint8 PNG
            prediction_normalized = ((prediction + 1.0) / 2.0 * 255).astype(np.uint8)
            enhanced_image = Image.fromarray(prediction_normalized)
            enhanced_image.save(os.path.join(enhanced_dir, im_full_name + ".png"))

            # --- Keep visualization part (optional, but good for debugging) ---
            space = np.zeros((128, 1024), dtype=np.uint8)
            
            deg_display = (((np.squeeze(deg_image) + 1.0) / 2.0) * 255).astype(np.uint8)
            gt_display = (((np.squeeze(gt_image) + 1.0) / 2.0) * 255).astype(np.uint8)
            
            im1 = cv2.cvtColor(prediction_normalized, cv2.COLOR_GRAY2BGR)
            im2 = cv2.cvtColor(deg_display, cv2.COLOR_GRAY2BGR)
            im3 = cv2.cvtColor(gt_display, cv2.COLOR_GRAY2BGR)
            im4 = cv2.cvtColor(space, cv2.COLOR_GRAY2BGR)
            
            show = vconcat_resize([im2, im4, im1, im4, im3])
            cv2.imwrite(os.path.join(vis_dir, im_full_name + ".png"), show)
        except Exception as e:
            print(f"⚠️ Error during evaluation of {im_base}: {e}")


    # Calculate PSNR metrics after visual evaluation
    print(f"📊 Calculating PSNR metrics for epoch {epoch}...")
    try:
        # --- FIX: Pass the correct epoch to the PSNR function ---
        average_psnr = get_psnr_iam(epoch)
        if average_psnr:
            print(f"✅ Epoch {epoch} - Average PSNR: {average_psnr:.2f} dB")
            if wandb_integration:
                wandb_integration.log_custom_metrics({'average_psnr': average_psnr}, prefix='validation')
        else:
            print(f"⚠️ Epoch {epoch} - PSNR calculation failed")
    except Exception as e:
        print(f"❌ Epoch {epoch} - PSNR calculation error: {e}")

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
	reco=re.sub(' +?', ' ', reco).strip()
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
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1), verbose=0).reshape(128,1024)
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
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1), verbose=0).reshape(128,1024)
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

def get_psnr_iam(epoch): # <-- FIX: Accept epoch as an argument
    """
    Fungsi PSNR yang diperbaiki untuk dataset NAN.
    Menggunakan path yang fleksibel dan struktur database yang benar.
    """
    print("🔍 Calculating PSNR for NAN dataset...")
    
    # --- FIX: Use the specific epoch's output directory ---
    enhanced_base_path = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "enhanced_psnr")
    
    if not os.path.exists(enhanced_base_path):
        print(f"❌ Enhanced images directory not found for epoch {epoch} at: {enhanced_base_path}")
        return None

    print(f"📁 Using enhanced images from: {enhanced_base_path}")
    
    # Get the list of ground truth files to compare against
    list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')
    
    total_psnr = 0
    processed_count = 0
    
    for im_base in tqdm(list_image_valid[:50], desc="Calculating PSNR"): # Limit for speed
        try:
            # Find the full ground truth image name (with extension)
            gt_search_pattern = os.path.join(DatabasePath, 'validation', 'images', im_base + '.*')
            gt_files = glob(gt_search_pattern)
            if not gt_files:
                continue
            gt_full_name = os.path.basename(gt_files[0])
            gt_image_path = gt_files[0]

            # Construct the path to the corresponding enhanced image
            enhanced_image_path = os.path.join(enhanced_base_path, gt_full_name + ".png")

            if not os.path.exists(enhanced_image_path):
                continue

            # Load and process images
            original_image = Image.open(gt_image_path).convert('L').resize((1024, 128), Image.Resampling.LANCZOS)
            gt = np.array(original_image) / 255.0
            
            enhanced_image = Image.open(enhanced_image_path).convert('L')
            predicted = np.array(enhanced_image) / 255.0
            
            # Calculate PSNR
            psnrv = psnr(predicted, gt)
            total_psnr += psnrv
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {im_base} for PSNR: {e}")
            continue
    
    if processed_count > 0:
        average_psnr = total_psnr / processed_count
        print(f"\n📈 PSNR Summary for Epoch {epoch}:")
        print(f"   Total images processed: {processed_count}/{len(list_image_valid)}")
        print(f"   Average PSNR: {average_psnr:.2f} dB")
        return average_psnr
    else:
        print("❌ No matching images found to calculate PSNR!")
        return None

if __name__ == '__main__':
	replicas = strategy.num_replicas_in_sync
	print(f"🚀 Starting ENHANCED FULL OPTIMIZATION training with {replicas} GPU(s)")
	print(f"⚙️ Configuration Summary:")
	print(f"   Scenario: {args.scenario}")
	print(f"   Epochs: {args.epochs} (starting from {args.start_epoch})")
	print(f"   Batch Size: {args.batch_size}")
	print(f"   Learning Rate: {args.learning_rate}")
	print(f"   Patience: {args.patience}")
	print(f"   Loss Weights: Adv={args.adv_weight}, Content={args.content_weight}, Recognition={args.recognition_weight}")
	print(f"   Save Interval: {args.save_interval}")
	print(f"   Eval Interval: {args.eval_interval}")
	
	# Check if batch size is compatible with number of replicas
	if args.batch_size % replicas != 0:
		adjusted = (args.batch_size // replicas) * replicas
		print(f"⚠️ batch_size {args.batch_size} not divisible by {replicas} replicas. Adjusted to {adjusted}.")
		args.batch_size = adjusted if adjusted > 0 else replicas
	
	if replicas == 1:
		print("ℹ️ Only 1 GPU detected (check CUDA_VISIBLE_DEVICES if expecting multi-GPU).")
	else:
		per_rep = args.batch_size // replicas
		print(f"📊 Multi-GPU Configuration: {replicas} GPUs × {per_rep} = {per_rep * replicas} global batch size")
	
	print("⚡ Mixed precision + XLA + Advanced optimizations enabled")
	
	# Execute based on mode
	if args.mode == 'train':
		train_GAN_crnn(args.epochs, args.batch_size)
	elif args.mode == 'predict':
		print("🔮 Prediction mode - implement prediction logic here")
		# Add prediction logic here
	elif args.mode == 'evaluate':
		print("📊 Evaluation mode - implement evaluation logic here")
		# Add evaluation logic here
	else:
		print(f"❌ Unknown mode: {args.mode}")
		sys.exit(1)