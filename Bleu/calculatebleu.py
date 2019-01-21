#-*- coding:utf-8 -*-
import sys
import codecs
import os
import math
import operator
import json
import sys
from functools import reduce
import fnmatch
import os



def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            files = fnmatch.filter(files,'*.txt')
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        #print si
        ref_counts = []
        ref_lengths = []
        #print references
        # Build dictionary of ngram counts
        for reference in references:
            #print 'reference' + reference
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    
    return bp


def geometric_mean(precisions,weight):
    sum = 0
    for i in range(4):
        if (precisions[i] == 0):
            sum += -9999999
        else:
            sum += weight[i]*math.log(precisions[i],math.e)
    return sum


def BLEU(candidate, references,weight):
    precisions = []
    bp = ''
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
        #print ('P'+str(i+1), ' = ',round(pr, 2))
    #print ('BP = ',round(bp, 2))
    bleu = math.exp(geometric_mean(precisions,weight)) * bp
    return bleu,precisions

if __name__ == "__main__":
    candidate, references = fetch_data('candidate.txt', 'testSet')
    bleu1,_ = BLEU(candidate, references ,[1,0,0,0])
    bleu2,_ = BLEU(candidate, references, [0.5,0.5,0,0])
    bleu3,_ = BLEU(candidate, references, [0.333,0.333,0.333,0])
    bleu4,precisions = BLEU(candidate, references, [0.25,0.25,0.25,0.25])
    for i in range(4):
        print('P'+str(i+1)+' = %f' %precisions[i])
    print('BLEU1 = ', round(bleu1, 4))
    print('BLEU2 = ', round(bleu2, 4))
    print('BLEU3 = ', round(bleu3, 4))
    print('BLEU4 = ', round(bleu4, 4))

    out = open('bleu_out.txt', 'w')
    out.write(str(bleu1))
    out.write(str(bleu2))
    out.write(str(bleu3))
    out.write(str(bleu4))
    out.close()