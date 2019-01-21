from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from keras.models import load_model
from keras.models import model_from_yaml
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from numpy import argmax
import pickle

def get_startsen_sequence(tokenizer):
	starseq = tokenizer.texts_to_sequences(['STARTSEN'])[0]
	starseq = pad_sequences([starseq], maxlen=38)
	return starseq

def generate_caption(img_features,tokenizer,model):
	input2 = np.asarray([img_features])
	input1 = get_startsen_sequence(tokenizer)
	caption = ['startsen']
	optword = ''

	count = 0
	while optword != 'endsen' and count != 38:
		count += 1
		optword = model.predict([input1, input2])
		optword = argmax(optword)
		for word, index in tokenizer.word_index.items():
			if index == optword:
				optword = word
				caption.append(optword)
				str = ' '.join(caption)
				input1 = tokenizer.texts_to_sequences([str])
				input1 = pad_sequences(input1, maxlen=38)
				break
	return ' '.join(caption)

def generate_candidate_n_save(filename,test_doc,IMGFeatures,model):

	candidate = []
	with open('Tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)
	for key, value in test_doc.items():
		candidate.append(generate_caption(IMGFeatures[key], tokenizer, model))

	with open(filename, 'wb') as f:
		pickle.dump(candidate,f)

	return candidate
def main():
	with open('vgg16_features.pkl','rb') as f:
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
	modelfile = os.path.join('MODELS','archi.txt')
	with open(modelfile, 'r') as f:
		model = f.read()
	model = model_from_yaml(model)
	model.load_weights(os.path.join('MODELS','weight002-val_loss3.249.h5'))


	#generate test candidate caption
	candidate = generate_candidate_n_save('candidate.pkl',test_doc,IMGFeatures,model)


	# with open('candidate.pkl','rb') as f:
	# 	candidate = pickle.load(f)


	#get test reference caption
	reference = []
	for key,value in test_doc.items():
		value = [' '.join(index.split()[1:-1]) for index in value]
		reference.append(value)

	# evaluate the bleu score
	print(corpus_bleu(reference,candidate,weights=[1,0,0,0]))
	print(corpus_bleu(reference,candidate,weights=[0.5,0.5,0,0]))
	print(corpus_bleu(reference,candidate,weights=[0.33,0.33,0.33,0]))
	print(corpus_bleu(reference,candidate,weights=[0.25,0.25,0.25,0.25]))
if __name__ == '__main__':
	with open('Tokenizer.pkl','rb') as f:
		h = pickle.load(f)
	main()
