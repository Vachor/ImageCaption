from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from keras.models import load_model
from keras.models import model_from_yaml
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from numpy import argmax
import pickle
from tensor_model import get_input_,get_max_sen_len
import tensorflow as tf

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def get_startsen_sequence(tokenizer):
	starseq = tokenizer.texts_to_sequences(['STARTSEN'])[0]
	starseq = pad_sequences([starseq], maxlen=38)
	return starseq
def idx_to_word(optword,tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == optword:
			optword = word
			return word
def beam_search():
	g = 5
def generate_caption(img_features,tokenizer,sess,key):
	input2 = np.asarray([img_features])
	caption = ['startsen']
	optword = ''

	str = ' '.join(caption)
	input1 = tokenizer.texts_to_sequences([str])
	input1 = pad_sequences(input1, maxlen=38)

	optword_tensor = sess.graph.get_tensor_by_name('Dense3_1/add:0')
	x1 = sess.graph.get_tensor_by_name('INPUT_1/Placeholder:0')
	x2 = sess.graph.get_tensor_by_name('INPUT_1/Placeholder_1:0')

	count = 0
	while optword != 'endsen' and count<38:
		count += 1
		optword = sess.run(optword_tensor,feed_dict={x1:input1,x2:input2})
		optword = argmax(optword)
		optword = idx_to_word(optword,tokenizer)
		caption.append(optword)
		str = ' '.join(caption)
		input1 = tokenizer.texts_to_sequences([str])
		input1 = pad_sequences(input1, maxlen=38)
	return caption[1:-1]


def generate_candidate_n_save(filename,test_doc,IMGFeatures,sess):

	candidate = []
	with open('Tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)
	for key, value in test_doc.items():
		candidate.append(generate_caption(IMGFeatures[key], tokenizer, sess,key))

	with open(filename, 'wb') as f:
		pickle.dump(candidate,f)

	return candidate
def main():
	with open('random_fet.pkl','rb') as f:
		IMGFeatures = pickle.load(f)

	#generate test document
	test_title = []
	with open(os.path.join('Flickr8k_text','Flickr_8k.testImages.txt'),'r') as f:
		for i in f:
			test_title.append(i.split('.')[0])

	with open('doc.pkl','rb') as f:
		doc = pickle.load(f)
	test_doc = {key:doc[key] for key in test_title}

	# load model
	# modelfile = os.path.join('MODELS','archi.txt')
	# with open(modelfile, 'r') as f:
	# 	model = f.read()
	# model = model_from_yaml(model)
	# model.load_weights(os.path.join('MODELS','weight004-val_loss3.441.h5'))

	# load model using saver
	sess = tf.InteractiveSession()
	imported_meta = tf.train.import_meta_graph('RNNModel/mnist.ckpt-3.meta')
	imported_meta.restore(sess, tf.train.latest_checkpoint('RNNModel/'))

	#generate test candidate caption
	candidate = generate_candidate_n_save('candidate.pkl',test_doc,IMGFeatures,sess)


	# with open('candidate.pkl','rb') as f:
	# 	candidate = pickle.load(f)


	#get test reference caption
	reference = []
	for key,value in test_doc.items():
		value = [index.split()[1:-1] for index in value]
		reference.append(value)
	with open('candidate.txt','w') as f:
		for i,value in enumerate(candidate):
			if i == 999:
				f.write(' '.join(value))
			else:
				f.write(' '.join(value)+'\r\n')

	f1 = open('reference1.txt', 'w')
	f2 = open('reference2.txt', 'w')
	f3 = open('reference3.txt', 'w')
	f4 = open('reference4.txt', 'w')
	f5 = open('reference5.txt', 'w')
	for i,value in enumerate(reference):

		for j,value2 in enumerate(value):
			if j == 0:
				if i == 999:
					temp1 = ' '.join(value2)
				else:
					temp1 = ' '.join(value2)+'\r\n'
				f1.write(temp1)
			elif j == 1:
				if i == 999:
					temp2 = ' '.join(value2)
				else:
					temp2 = ' '.join(value2)+'\r\n'
				f2.write(temp2)
			elif j == 2:
				if i == 999:
					temp3 = ' '.join(value2)
				else:
					temp3 = ' '.join(value2)+'\r\n'
				f3.write(temp3)
			elif j == 3:
				if i == 999:
					temp4 = ' '.join(value2)
				else:
					temp4 = ' '.join(value2)+'\r\n'
				f4.write(temp4)
			elif j == 4:
				if i == 999:
					temp5 = ' '.join(value2)
				else:
					temp5 = ' '.join(value2)+'\r\n'
				f5.write(temp5)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
	# evaluate the bleu score
	bleu1 = corpus_bleu(reference, candidate, weights=[1, 0, 0, 0])
	print(bleu1)
	bleu2 = corpus_bleu(reference,candidate,weights=[0.5,0.5,0,0])
	print(bleu2)
	bleu3 = corpus_bleu(reference,candidate,weights=[0.33,0.33,0.33,0])
	print(bleu3)
	bleu4 = corpus_bleu(reference,candidate,weights=[0.25,0.25,0.25,0.25])
	print(bleu4)
	return bleu1,bleu2,bleu3,bleu4
if __name__ == '__main__':
	with open('Tokenizer.pkl','rb') as f:
		h = pickle.load(f)
	main()
