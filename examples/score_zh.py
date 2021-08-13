#!/usr/bin/env python
OOV =['SICLE', '勒萌萌']
import os
import kenlm
import sys
import numpy as np
#sys.path.append("../")
sys.path.append("/Users/qzlbxyz/PycharmProjects/pycorrector/")
sys.path.append("/data/liubin/pycorrector/")
# print(sys.path)
import pycorrector_adjust
from tqdm import tqdm
from pycorrector_adjust import Corrector
import math

# Dontla：定义sigmoid函数
def sigmoid(inx):
    if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))

model_type = sys.argv[2]
if model_type == '1':
    LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm')
elif model_type == '2':
    LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/mix_bjh_order_3gram_large.klm')
elif model_type == '3':
    # LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/baijiahao_3order_punctuation_ori.klm')
    LM = os.path.join('/home/liubin/.pycorrector/datasets/baijiahao_3order_punctuation_ori.klm')
else:
    LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/mix_orders_3order.klm')
model = kenlm.LanguageModel(LM)
corrector = Corrector(language_model_path=LM, custom_word_freq_path='../data/lex')
print('{0}-gram model'.format(model.order))

# Check that total full score = direct score
def logppl2score(logppl):
    '''
    多数分数集中在2.0，若将一般分设定为80分上下，则涉及如下：
    logppl：0-3      3-6
    score: 100-70   70-0
    '''
    if logppl < 0:
        return 100
    elif logppl > 6:
        return 0
    elif 0 <= logppl <= 3:
        return int((3 - logppl) * (30 / 3) + 60)
    else:
        return int((6 - logppl) * (70 / 3))


def ppl2score(ppl):
    if ppl <= 100:
        return int((100 - ppl) / 10 + 90)
    elif 100 < ppl <= 300:
        return int((300 - ppl) / 20 + 80)
    elif 300 < ppl <= 700:
        return int((700 - ppl) / 40 + 70)
    elif 700 < ppl <= 1500:
        return int((1500 - ppl) / 80 + 60)
    elif 1500 < ppl <= 7500:
        return int((7500 - ppl) / 100)
    else:
        return 0


pycorrector_adjust.set_log_level('INFO')
if __name__ == '__main__':
    test_file = open(sys.argv[1], 'r')
    totol_pplscores = []
    totol_logpplscores = []
    totol_logppl = []
    totol_ppl = []
    oov_count = 0
    good_ppl = 0
    for line in tqdm(test_file.readlines()):
        sentence = line.strip()
        # print("<<<<<< sentence is: {}".format(sentence))
        logprob = model.score(sentence)
        words = sentence.split(' ')
        ppl = pow(10, -1.0/(1+len(words))*(logprob))
        logppl = round(math.log10(ppl), 1)
        # print('score={}'.format(ppl2score(ppl)))
        totol_pplscores.append(ppl2score(ppl))
        totol_logpplscores.append(logppl2score(logppl))
        totol_logppl.append(logppl)
        totol_ppl.append(ppl)
        # print('ppl={}'.format(ppl))
        if ppl <= 300:
            good_ppl+=1
            print("<<<<<< sentence is: {}".format(sentence))
            print('score={}'.format(ppl2score(ppl)))
            print('ppl={}'.format(ppl))

            words = ['<s>'] + sentence.split() + ['</s>']
            for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
                if oov:
                    print('\t"{0}" is an OOV'.format(words[i+1]))
                    oov_count+=1
        # idx_errors = corrector.detect(''.join(sentence.split()))
        # print(sentence, '>>>>>>', idx_errors)
        # print()


    print(len(totol_ppl))
    print('avg of ppl scores: ', np.mean(totol_pplscores))
    print('avg of logppl scores: ', np.mean(totol_logpplscores))
    print('avg of ppl: ', np.mean(totol_ppl))
    print('avg of logppl: ', np.mean(totol_logppl))

    print(oov_count)
    print(good_ppl)
