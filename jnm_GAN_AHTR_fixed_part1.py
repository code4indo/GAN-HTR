import os
os.environ["PYTHONIOENCODING"] = "utf-8"
#1 geforce
#0 titan
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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

import math
import tensorflow as tf
 

from PIL import Image
from tqdm import tqdm
import random
import sys
import codecs
import re
import cv2
import tqdm
from glob import glob
from tqdm import tqdm
from data import preproc as pp


##########################################################################################################
##########################################################################################################
##########################################################################################################
rootPath='./'
DatabasePath='datasets/nan_raw_biner/'
scenario='S_nan_OP'

num_classes = 2

depth = 5
width = 1

size=(128,1024, 1)
input_size = (128,1024, 1)
input_size_crnn = (1024,128, 1)
max_text_length=128
divider=4 
charset_base = read_file_char(rootPath+ 'Sets/CHAR_LIST')
f=open(rootPath+ 'Sets/charset_base.txt','w+', encoding='utf-8')
f.writelines(charset_base)
f.close()

def read_file_shuffle(filename):
	# #######################################
	# f= open(filename, 'r')
	# lines=[]
	# for line in f:
	# 	line=line.rstrip()
	# 	if len(line)>0:
	# 		lines.append(line)
	# #random.shuffle(lines)
	# return lines
	
	lines=[]
	f=codecs.open(filename, 'r','utf-8')
	for line in f:
		line=line.rstrip()
		if len(line)>0:
			lines.append(line)
	random.shuffle(lines)
	return lines
def read_file(filename):
	lines=[]
	f=codecs.open(filename, 'r','utf-8')
	for line in f:
		line=line.rstrip()
		if len(line)>0:
			lines.append(line)
	return lines
def read_file_char(filename):
	lines=[]
	f=codecs.open(filename, 'r','utf-8')
	for line in f:
		line=line.rstrip()
		if len(line)>0:
			lines.append(line)
	return lines

def normalizeTranscription(text):
	# normalize text
	text=text.lower()
	return text
	

def unet_generator():
	
	inputs = Input(shape=(128,1024, 1))   ##### degraded image 
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	bn = BatchNormalization()(conv1)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	bn = BatchNormalization()(conv2)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	bn = BatchNormalization()(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	bn = BatchNormalization()(conv4)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv4)
	drop4 = Dropout(0.5)(bn)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	bn = BatchNormalization()(conv5)
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv5)
	drop5 = Dropout(0.5)(bn)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
	bn = BatchNormalization()(up6)
	
	merge6 = concatenate ([drop4, bn])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	bn = BatchNormalization()(conv6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization()(up7)
	merge7 = concatenate ([conv3, bn])
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	bn = BatchNormalization()(conv7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv7)

	
	
	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization()(up8)
	merge8 = concatenate ([conv2, bn])
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	bn = BatchNormalization()(conv8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization()(up9)
	
	merge9 = concatenate ([conv1, bn])
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	bn = BatchNormalization()(conv9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization()(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(bn)

	model = Model(inputs=inputs, outputs=conv10)
	
	return model	


def optimizer():
	return Adam(learning_rate=1e-4)


def discriminator_patch():

	img_A = Input(shape=(128,1024, 1))
	img_B = Input(shape=(128,1024, 1))

	# Concatenate image and conditioning image by channels to produce input
	combined_imgs = Concatenate(axis=-1)([img_A, img_B])

	d1 = Conv2D(64, 4, strides=2, padding='same')(combined_imgs)
	d1 = LeakyReLU(negative_slope=0.2)(d1)

	d2 = Conv2D(128, 4, strides=2, padding='same')(d1)
	d2 = BatchNormalization(momentum=0.8)(d2)
	d2 = LeakyReLU(negative_slope=0.2)(d2)

	d3 = Conv2D(256, 4, strides=2, padding='same')(d2)
	d3 = BatchNormalization(momentum=0.8)(d3)
	d3 = LeakyReLU(negative_slope=0.2)(d3)

	d4 = Conv2D(512, 4, padding='same')(d3)
	d4 = BatchNormalization(momentum=0.8)(d4)
	d4 = LeakyReLU(negative_slope=0.2)(d4)

	validity = Conv2D(1, 4, padding='same')(d4)

	discriminator = Model([img_A, img_B], validity)

	discriminator.compile(loss='mse', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

	return discriminator

def ctc_loss_lambda_func(y_true, y_pred):
	
	if len(y_true.shape) > 2:
		y_true = tf.squeeze(y_true)

	# y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
	# output of every model is softmax
	# so sum across alphabet_size_1_hot_encoded give 1
	#               string_length give string length
	input_length = tf.math.reduce_sum(y_pred, axis=2, keepdims=False)
	input_length = tf.math.reduce_sum(input_length, axis=1, keepdims=True)

	# y_true strings are padded with 0
	# so sum of non-zero gives the length of the string
	label_length = tf.math.reduce_sum(tf.cast(y_true, tf.float32), axis=1, keepdims=True)

	loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

	# remove nan for safety
	loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)

	return loss

def flor(input_size, num_classes):
	input_data = Input(name='input', shape=input_size, dtype='float32')
	
	# make cnn
	inner = Dense(num_classes + 1, activation='softmax', name='dense2')(input_data)
	
	# prediction model is used to predict classes of inputs
	prediction_model = Model(inputs=input_data, outputs=inner)
	
	inputs = Input(name='inputs', shape=input_size, dtype='float32')
	y_true = Input(name='y_true', shape=[max_text_length], dtype='float32')

	outputs = prediction_model(inputs)
	loss_func = Lambda(ctc_loss_lambda_func, output_shape=(1,), name='ctc')([y_true, outputs])

	model = Model(inputs=[inputs, y_true], outputs=loss_func)

	return inputs, outputs

def flor_discriminator():
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	model = Model(inputs=inputs, outputs=outputs)
	
	return model	

def flor_discriminator_full():
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	model = Model(inputs=inputs, outputs=outputs)
	
	return model

def readGrayPair(im_name, split='train'):
	deg_image_path = os.path.join('datasets/nan_distorted/', split, im_name)

	original_image = Image.open(deg_image_path)
	original_image = original_image.resize((1024,128), Image.LANCZOS)
	grey_image = original_image.convert('L')
	
	grey_image.save("deg_image2.png")
	deg_image = plt.imread("deg_image2.png")
	
	gt_image_path = os.path.join(DatabasePath, split, 'images', im_name)
	original_image = Image.open(gt_image_path)
	original_image = original_image.resize((1024,128), Image.LANCZOS)
	grey_image = original_image.convert('L')
	grey_image.save("gt_image2.png")
	gt_image = plt.imread("gt_image2.png")
	
	return deg_image, gt_image

def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC): 
	# take minimum width 
	w_min = min(img.shape[1] for img in img_list) 
	
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
	gan = Model([gan_input], [out_discrimintor_1, out_generator,  out_discrimintor_2])

	gan.compile(loss=['mse','binary_crossentropy',ctc_loss_lambda_func], loss_weights=[1,10,1], optimizer=optimizer)   ##### the weight are to discuss later Please dont forget !!!
	return gan

def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		index = charset_base.index(item)
		encoded.append(index)
	
	# encoded=encoded[::-1]  ############this is done only for arabic, otherwise remove this line
	return encoded
