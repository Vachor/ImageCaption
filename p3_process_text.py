import string
import os
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
import scipy.io as sio

def construct_wash_doc(filename):
	punctuation = np.zeros(32)
	total = 0

	doc = {}
	table = str.maketrans('','',string.punctuation)
	print(string.punctuation)
	#load all the text,construct a list in dictionary, and wash all the caption
	with open(filename) as f:
		line = f.readline()
		while line:
			#load the txt data line by line, filter some punctuation, and make all word in text to lower characters
			key = (line.split()[0]).split('.')[0]
			text = line.split()[1:]

			for index_puc,value in enumerate(string.punctuation):
				if(value in text):
					punctuation[index_puc] += text.count(value)
			total += len(text)

			text.insert(0,'startsen')
			text.insert(len(text),'endsen')
			text = ' '.join(text)
			text = text.translate(table)
			text = text.split()
			text = ' '.join(text)
			text = text.lower()

			#and make a dictionary to store the caption
			if key not in doc:    #if the key is not in the dictionary, construct it, and make a list
				doc[key] = []
			doc[key].append(text)
			line = f.readline()
	total = np.float32(total)
	sio.savemat('wordcount.mat', {'punctuation': punctuation, 'total':total,'str':string.punctuation})
	return doc
def construct_tokenizer(doc,trainlist):
	tokenizer = Tokenizer()
	alltext = ''
	for key in trainlist:
		for i in doc[key]:
			alltext += (i+' ')
	tokenizer.fit_on_texts([alltext])
	return tokenizer
def get_trainfiles_list(filepath):
	trainlist = []
	with open(filepath,'r') as f:
		line = f.readline()
		while line:
			line = line.split('.')[0]
			trainlist.append(line)
			line = f.readline()
	return trainlist

def main():
	#construct caption documentary
	print('constructing a caption documentary...')
	filename = os.path.join('Flickr8k_text','Flickr8k.token.txt')
	doc = construct_wash_doc(filename)

	#construct tokenizer
	print('constructing a tokenizer...')
	filepath = os.path.join('Flickr8k_text','Flickr_8k.trainImages.txt')
	trainlist = get_trainfiles_list(filepath)

	# with open('trainimages.txt','w') as f:
	# 	for i in trainlist:
	# 		for j in doc[i]:
	# 			j = j.split(' ')
	# 			j = j[1:-1]
	# 			j = ' '.join(j)
	# 			f.write(j+'\n')


	tokenizer = construct_tokenizer(doc,trainlist)

	#save tokenizer and documentary
	print('save tokenizer and documentary...')
	with open('Tokenizer.pkl','wb') as f:
		pickle.dump(tokenizer,f)
	with open('doc.pkl','wb') as f:
		pickle.dump(doc,f)
	print('all done')
if __name__ == '__main__':
    main()