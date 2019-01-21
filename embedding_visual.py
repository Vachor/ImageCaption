import pickle
import os
import numpy as np

with open('Tokenizer.pkl', 'rb') as f:
	tokenizer = pickle.load(f)
with open('doc.pkl', 'rb') as f:
	doc = pickle.load(f)
max_voc_plus_1 = len(tokenizer.word_index) + 1  # plus_one is for embedding layers
 # get the maximam length of sentences in all captions

word_index = tokenizer.word_index
embedding_index = {}

# make a embedding matrix
with open(os.path.join('glove.6B', 'glove.6B.50d.txt'), 'r', encoding='utf-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = (values[1:])
		embedding_index[word] = coefs
embedding_dic = {}
with open('word2vec.txt','w') as f:
	for word, i in word_index.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			newcontent = word + ' ' + ' '.join(embedding_vector) + '\n'
			f.write(newcontent)