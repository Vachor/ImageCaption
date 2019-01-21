import pickle

with open('Tokenizer.pkl', 'rb') as f:
	tokenizer = pickle.load(f)
h = tokenizer.word_index
with open('v1.txt','w') as f:
	f.write(h)