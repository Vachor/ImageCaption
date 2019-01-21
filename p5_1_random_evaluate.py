from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from keras.models import load_model
from keras.models import model_from_yaml
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from numpy import argmax
from p4_4_model import get_max_sen_len
import pickle
import random

def get_index_word(Tokenizer):
	word_index = Tokenizer.word_index
	index_word = {value:key for key,value in word_index.items()}
	return index_word

def get_startsen_sequence(tokenizer):
	starseq = tokenizer.texts_to_sequences(['STARTSEN'])[0]
	starseq = pad_sequences([starseq], maxlen=38)
	return starseq

def generate_candidate(test_doc):

	candidate = []
	with open('Tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)

	max_sen_len = get_max_sen_len(test_doc)
	str = ''
	index_word = get_index_word(tokenizer)
	for key,value in test_doc.items():
		for i in range(0,max_sen_len):
			rad = random.randint(1,len(index_word))
			str += index_word[rad] + ' '
		str = str[:-1]
		candidate.append(str)
	return candidate
def main():
	#generate test document
	test_title = []
	with open(os.path.join('Flickr8k_text','Flickr_8k.testImages.txt'),'r') as f:
		for i in f:
			test_title.append(i.split('.')[0])

	with open('doc.pkl','rb') as f:
		doc = pickle.load(f)
	test_doc = {key:doc[key] for key in test_title}



	#generate test candidate caption
	candidate = generate_candidate(test_doc)


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
	main()
