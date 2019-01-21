from scipy.misc import imread,imresize
from keras.models import load_model,model_from_yaml
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
import os
import pickle
import numpy as np
import tensorflow as tf
from p5_evaluate_tensor import generate_caption
from keras.preprocessing.text import Tokenizer
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

def get_IMG_features(img):
	cnnmodel = load_model(os.path.join('CNNMODELS','ResNet50.h5'))
	img = imread(img)
	img = imresize(img,[224,224,3])
	features = cnnmodel.predict(np.asarray([img]))
	return features[0]

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def main():
	with open('vgg16_features.pkl','rb') as f:
		features = pickle.load(f)
	with open('Tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)

	#load model using yaml
	imported_meta = tf.train.import_meta_graph('model/mnist.ckpt-1.meta')
	imported_meta.restore(sess, tf.train.latest_checkpoint('model/'))

	# load model using load_model
	#model = load_model(os.path.join('model','ep002-val_loss3.222.h5'))

	# get image features
	img_features = get_IMG_features('example2.jpg')
	# generate captions
	caption = generate_caption(img_features, tokenizer,sess)
	print(caption)
if __name__ == '__main__':
    main()