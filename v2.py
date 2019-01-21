from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from Bleu.calculatebleu import BLEU,fetch_data
import os
import numpy as np
BLEU_all = np.zeros([4,19])
count = 0
candidate_all, references = fetch_data(os.path.join('Bleu','candidate.txt'), os.path.join('Bleu','testSet'))
for i,candidate in enumerate(candidate_all):
	Blue1,_ = BLEU([candidate],references, [1, 0, 0, 0])
	Blue2,_ = BLEU([candidate],references, [0.5, 0.5, 0, 0])
	Blue3,_ = BLEU([candidate],references, [0.33, 0.33, 0.33, 0])
	Blue4,_ = BLEU([candidate],references, [0.25, 0.25, 0.25, 0.25])
	BLEU_all[0,count] = Blue1
	BLEU_all[1,count] = Blue2
	BLEU_all[2,count] = Blue3
	BLEU_all[3,count] = Blue4
	count += 1
h = 5