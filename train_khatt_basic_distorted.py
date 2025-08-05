import os
os.environ["PYTHONIOENCODING"] = "utf-8"
#1 geforce
#0 titan
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

import tensorflow as tf
import argparse
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
import string

# define parameters
source = "khatt"
arch = "flor" ########ne pas modifier, nous utilisons architeture crnn de flor

# define input size, number max of chars per line and list of valid chars
max_text_length = 128  ####not change this value
img_width=1024 #########for crnn
img_height=128 #########for crnn
input_size_crnn = (1024,128, 1)
input_size = (128,1024, 1) #############for the GAN


def get_callbacks(logdir, checkpoint, monitor="val_loss", verbose=1):
        """Setup the list of callbacks for the model"""

        callbacks = [

          CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=15,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=20,
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

charset_base=read_file_char('Sets/CHAR_LIST')

def ctc_loss_lambda_func(y_true, y_pred):
	"""Function for computing the CTC loss"""

	if len(y_true.shape) > 2:
		y_true = tf.squeeze(y_true)

	input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
	input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
	label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")
	loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
	loss = tf.reduce_mean(loss)

	return loss

def build_crnn():
	from network.model import flor
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)
	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
	return model	

def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		index = charset_base.index(item)
		encoded.append(index)
	encoded=encoded[::-1]
	return encoded

def train_crnn(crnn, dataset_path, output_path, ep_start=0, epochs=130, batch_size=32):
	target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
	os.makedirs(output_path, exist_ok=True)

	list_image_train = read_file_shuffle('Sets/list_train')
	list_image_valid = read_file_shuffle('Sets/list_valid')
	list_lines = read_file('Sets/lines.txt')

	x_train_rcnn=[]
	y_train_rcnn=[]
	for im in tqdm(list_image_train, desc="Loading Train Set"):
		matched_lines = [s for s in list_lines if im in s]
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		line = normalizeTranscription(text_line)
		len_trancription=len(line.split())
		if len_trancription < max_text_length :
			img_path = os.path.join(dataset_path, im + '.tif')
			if not os.path.exists(img_path):
				continue
			imgx=pp.preprocess(img_path, input_size_crnn)
			x_train_rcnn.append(imgx)
			encoded_txt=encode_txt(line)
			y_train_rcnn.append(encoded_txt)

	y_train_rcnn = [np.pad(y, (0, max_text_length - len(y))) for y in y_train_rcnn]
	y_train_rcnn = np.asarray(y_train_rcnn, dtype=np.int16)
	x_train_rcnn = np.asarray(x_train_rcnn)
	x_train_rcnn = pp.normalization(x_train_rcnn)

	x_valid_rcnn=[]
	y_valid_rcnn=[]
	for im in tqdm(list_image_valid, desc="Loading Validation Set"):
		matched_lines = [s for s in list_lines if im in s]
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		line = normalizeTranscription(text_line)
		len_trancription=len(line.split())
		if len_trancription < max_text_length :
			img_path = os.path.join(dataset_path, im + '.tif')
			if not os.path.exists(img_path):
				continue
			img=pp.preprocess(img_path, input_size_crnn)
			x_valid_rcnn.append(img)
			encoded_txt=encode_txt(line)
			y_valid_rcnn.append(encoded_txt)

	y_valid_rcnn = [np.pad(y, (0, max_text_length - len(y))) for y in y_valid_rcnn]
	y_valid_rcnn = np.asarray(y_valid_rcnn, dtype=np.int16)
	x_valid_rcnn = pp.normalization(x_valid_rcnn)

	validation_data=(x_valid_rcnn,y_valid_rcnn)
	callbacks = get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)
	crnn.fit(x_train_rcnn,y_train_rcnn, validation_data=validation_data,batch_size=batch_size,initial_epoch=ep_start, epochs=epochs, verbose=1,
	 callbacks=callbacks,shuffle=True,validation_freq=1)
	return crnn

def ocr_crnn(filename,dtgen,model):
	text = ''
	input_size = (1024, 128, 1)

	im=pp.preprocess(filename,input_size)
	x_test = []
	x_test.append(im)
	x_test=pp.normalization(x_test)

	predicts, _ = model.predict(x=x_test,
							use_multiprocessing=False,
							ctc_decode=True,
							verbose=0)

	predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
	text=predicts[0]
	s=text.split()
	s=s[::-1]
	reco=' '.join(s)
	reco=reco.strip()
	return reco

def loadCRNNModel(output_path, batch_size):
	from data.generator import DataGenerator
	input_size = (1024, 128, 1)
	dtgen = DataGenerator(source="",
						  batch_size=batch_size,
						  charset=charset_base,
						  max_text_length=max_text_length)

	from network.model import HTRModel
	model = HTRModel(architecture=arch,
					 input_size=input_size,
					 vocab_size=dtgen.tokenizer.vocab_size,
					 beam_width=10,
					 stop_tolerance=20,
					 reduce_tolerance=15)

	model.compile(learning_rate=0.001)
	
	checkpoint_path = os.path.join(output_path, 'checkpoint_weights.hdf5')
	if os.path.exists(checkpoint_path):
		print(f"Loading checkpoint from: {checkpoint_path}")
		model.load_checkpoint(target=checkpoint_path)
	else:
		print(f"Warning: Checkpoint file not found at {checkpoint_path}. Model is initialized with random weights.")

	return dtgen,model

def recognition(dataset_path, output_path, batch_size, set_name):
	path_test = dataset_path
	list_lines = read_file('Sets/lines.txt')
	dtgen, model = loadCRNNModel(output_path, batch_size)
	list_image_valid = read_file('Sets/list_test')
	list_reco_c = []
	list_reco_w = []
	list_truth_c = []
	list_truth_w = []
	i=0
	for im in tqdm(list_image_valid, desc=f"Running Recognition on {set_name} set"):
		matched_lines = [s for s in list_lines if im in s]
		if not matched_lines:
			continue
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		text_line = normalizeTranscription(text_line)
		truth_char = text_line
		li = text_line.split()
		
		if len(li) < 128:
			img_path = os.path.join(path_test, im + '.tif')
			if not os.path.exists(img_path):
				continue
			
			gen_txt = ocr_crnn(img_path, dtgen, model)
			print(f"Image: {im}, Truth: {truth_char}, Recognized: {gen_txt}")
			list_reco_c.append(gen_txt + '\n')
			list_truth_c.append(truth_char + '\n')
			words = gen_txt.replace(' ', '').replace('sp', ' ')
			list_reco_w.append(words + '\n')
			twords = truth_char.replace(' ', '').replace('sp', ' ')
			list_truth_w.append(twords + '\n')

	path_result = os.path.join(output_path, set_name)
	os.makedirs(path_result, exist_ok=True)

	with codecs.open(os.path.join(path_result, f'c_reco_{set_name}.txt'), 'w', 'utf-8') as f:
		f.writelines(list_reco_c)
	with codecs.open(os.path.join(path_result, f'c_truth_{set_name}.txt'), 'w', 'utf-8') as f:
		f.writelines(list_truth_c)
	with codecs.open(os.path.join(path_result, f'w_reco_{set_name}.txt'), 'w', 'utf-8') as f:
		f.writelines(list_reco_w)
	with codecs.open(os.path.join(path_result, f'w_truth_{set_name}.txt'), 'w', 'utf-8') as f:
		f.writelines(list_truth_w)

	command_cer = f"wer -a -e {os.path.join(path_result, f'c_truth_{set_name}.txt')} {os.path.join(path_result, f'c_reco_{set_name}.txt')} > {os.path.join(path_result, f'evaluate_{set_name}_CER.txt')}"
	os.system(command_cer)
	command_wer = f"wer -a -e {os.path.join(path_result, f'w_truth_{set_name}.txt')} {os.path.join(path_result, f'w_reco_{set_name}.txt')} > {os.path.join(path_result, f'evaluate_{set_name}_WER.txt')}"
	os.system(command_wer)
	print(f"Results for {set_name} saved in {path_result}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the HTR model on the KHATT dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the root directory of the KHATT dataset images for training and default testing.")
    parser.add_argument("--dataset_dir_hard", type=str, help="Path to the 'hard' test set images. If not provided, this evaluation will be skipped.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("output-KHATT-distorted", source, arch), help="Directory to save logs, checkpoints, and results.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--do_train", action='store_true', help="Enable the training phase.")
    parser.add_argument("--do_eval", action='store_true', help="Enable the evaluation phase.")

    args = parser.parse_args()

    if args.do_train:
        print("\n--- Starting Training Phase ---")
        crnn = build_crnn()
        train_crnn(crnn,
                   dataset_path=args.dataset_dir,
                   output_path=args.output_dir,
                   epochs=args.epochs,
                   batch_size=args.batch_size)
        print("\n--- Training Finished ---")

    if args.do_eval:
        print("\n--- Starting Default Recognition Task ---")
        recognition(dataset_path=args.dataset_dir, output_path=args.output_dir, batch_size=args.batch_size, set_name="default")

        if args.dataset_dir_hard:
            print("\n--- Starting Hard Recognition Task ---")
            recognition(dataset_path=args.dataset_dir_hard, output_path=args.output_dir, batch_size=args.batch_size, set_name="hard")
        else:
            print("\n--- Skipping Hard Recognition Task (path not provided) ---")
    
    if not args.do_train and not args.do_eval:
        print("Please specify an action: --do_train or --do_eval")

if __name__ == '__main__':
	main()