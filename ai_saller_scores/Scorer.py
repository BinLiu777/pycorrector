import sys
sys.path.append("/Users/qzlbxyz/PycharmProjects/pycorrector/")
sys.path.append("/data/liubin/pycorrector/")
from fuzzywuzzy import fuzz
import jieba
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import torch
import os
from tqdm import tqdm
import pickle
import kenlm
from pycorrector_adjust import Corrector
from seq2vec.bert_vec import BertTextNet, BertSeqVec
import numpy as np
import re
import math
import pkuseg
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Scorer(object):
    def __init__(self, embedding_type='bert', device='cpu'):
        self.embedding_type = embedding_type
        self.init_score = 60
        self.add_score1 = 10
        self.add_score2 = 5
        self.delete_score = -15
        self.fuzzy_theshold_Levenshtein = 50
        self.fuzzy_theshold_tfidf = 0.8
        self.fuzzy_theshold_bert = 0.16
        self.device = torch.device(device)

        self.sim_words = {'你好': '您好'}

        self.init_lang_model()
        self.init_embedding_model(embedding_type)

        self.read_pp_files()

        self.seg2 = pkuseg.pkuseg(user_dict='../data/lex')

    def init_lang_model(self):
        # LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm')
        LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/mix_bjh_order_3gram_large.klm')
        # LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/baijiahao_3order_punctuation_ori.klm')
        # LM = os.path.join('/home/liubin/.pycorrector/datasets/baijiahao_3order_punctuation_ori.klm')
        # LM = os.path.join('/Users/qzlbxyz/.pycorrector/datasets/mix_orders_3order.klm')
        self.lang_model = kenlm.LanguageModel(LM)
        self.corrector = Corrector(language_model_path=LM, custom_word_freq_path='../data/lex')
        # print('{0}-gram model'.format(model.order))

    def init_embedding_model(self, embedding_type):
        if embedding_type == 'leven':
            pass
        elif embedding_type == 'tfidf':
            # vectorizer, transformer = self.train_tfidf_model()
            self.vectorizer, self.transformer = self.init_tfidf_model()
        elif embedding_type == 'bert':
            self.seq2vec = self.init_bert_model()
        else:
            raise NotImplementedError

    def read_pp_files(self):
        path1 = '../data/礼貌用语.txt'
        path2 = '../data/禁语.txt'

        self.polite_words = []
        self.forbidden_words = []
        self.polite_sents = []
        self.forbidden_sents = []
        self.polite_sent_vecs = []
        self.forbidden_sent_vecs = []

        with open(path1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) <= 4:
                    jieba.add_word(line.strip())
                    self.polite_words.append(line.strip())
                    continue
                line_cuts = jieba.lcut(line.strip())
                if len(line_cuts) == 1:
                    self.polite_words.append(line_cuts[0])
                elif len(line_cuts) > 1:
                    self.polite_sents.append(line.strip())
                    self.polite_sent_vecs.append(self.seq2vec.seq2vec(line.strip(), self.device))
                else:
                    continue

        with open(path2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) <= 4:
                    jieba.add_word(line.strip())
                    self.forbidden_words.append(line.strip())
                    continue
                line_cuts = jieba.lcut(line.strip())
                if len(line_cuts) == 1:
                    self.forbidden_words.append(line_cuts[0])
                elif len(line_cuts) > 1:
                    self.forbidden_sents.append(line.strip())
                    self.forbidden_sent_vecs.append(self.seq2vec.seq2vec(line.strip(), self.device))
                else:
                    continue

    # def train_tfidf_model(self):
    #     with open('../data/bert_train/train.txt', 'r') as f:
    #         lines = f.readlines()
    #     pure_saller_texts_cut = []
    #     for line in tqdm(lines):
    #         line = line.strip()
    #         line_cut = jieba.lcut(line)
    #         line_cut = [s for s in line_cut if s not in ['，', '。', '？', '！']]
    #         pure_saller_texts_cut.append(' '.join(line_cut))
    #
    #     vectorizer = CountVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b")
    #     transformer = TfidfTransformer()
    #     tfidf = transformer.fit_transform(vectorizer.fit_transform(pure_saller_texts_cut))
    #     word = vectorizer.get_feature_names()
    #     print("word feature length: {}".format(len(word)))
    #
    #     feature_path = '../models/tfidf/feature.pkl'
    #     with open(feature_path, 'wb') as fw:
    #         pickle.dump(vectorizer.vocabulary_, fw)
    #
    #     tfidftransformer_path = '../models/tfidf/tfidftransformer.pkl'
    #     with open(tfidftransformer_path, 'wb') as fw:
    #         pickle.dump(transformer, fw)
    #
    #     return vectorizer, transformer

    def init_tfidf_model(self):

        feature_path = '../models/tfidf/feature.pkl'
        vectorizer = CountVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b", decode_error="replace",
                                     vocabulary=pickle.load(open(feature_path, "rb")))
        tfidftransformer_path = '../models/tfidf/tfidftransformer.pkl'
        transformer = pickle.load(open(tfidftransformer_path, "rb"))
        word = vectorizer.get_feature_names()
        print("word feature length: {}".format(len(word)))
        return vectorizer, transformer

    def init_bert_model(self):
        # bert_path = '/data/liubin/language_model/models/bert/order_epoch15/'
        bert_path = '/Users/qzlbxyz/PycharmProjects/LangModel/models/bert/bert_zhijian/'

        text_net = BertTextNet(bert_path).to(self.device)
        seq2vec = BertSeqVec(text_net)  # 将模型实例给向量化对象
        return seq2vec

    def format_score(self, score):
        if score > 100:
            score = 100
        elif score < 0:
            score = 0
        return score

    def precision_match(self, text):
        score = self.init_score
        words = jieba.lcut(text)
        # print('分词后结果：', words)
        result = []
        keywords_count = dict()
        for word in words:
            if word in self.polite_words:
                if keywords_count.get(self.sim_words.get(word, word), 0) == 0:
                    result.append((word, self.add_score1))
                    keywords_count[self.sim_words.get(word, word)] = keywords_count.get(self.sim_words.get(word, word), 0) + 1
                elif keywords_count.get(self.sim_words.get(word, word), 0) == 1:
                    result.append((word, self.add_score2))
                    keywords_count[self.sim_words.get(word, word)] = keywords_count.get(self.sim_words.get(word, word), 0) + 1
            if word in self.forbidden_words:
                result.append((word, self.delete_score))
        for w, s in result:
            score += s
        score = self.format_score(score)
        return score, result

    def fuzzy_match_Levenshtein(self, text):
        score = self.init_score
        result = []
        for sent in self.polite_sents:
            s = fuzz.partial_ratio(text, sent)
            # print(text,sent,s)
            if s > self.fuzzy_theshold_Levenshtein:
                score = 100
                result.append((text, 100))
                break

        for sent in self.forbidden_sents:
            s = fuzz.partial_ratio(text, sent)
            # print(text, sent, s)
            if s > self.fuzzy_theshold_Levenshtein:
                if score == 100:
                    raise Exception('同时符合文明用语和服务禁语！')
                score = 0
                result.append((text, 0))
                break

        return score, result

    def fuzzy_match_tfidf(self, text):
        line_cut = [s for s in jieba.lcut(text) if s not in ['，', '。', '？', '！']]
        query_text = ' '.join(line_cut)
        query_vec_ = self.vectorizer.transform([query_text])
        query_vec = self.transformer.transform(query_vec_).toarray()[0]
        score = self.init_score
        result = []
        for sent in self.polite_sents:
            candi_text = ' '.join(jieba.lcut(sent))
            candi_vec_ = self.vectorizer.transform([candi_text])
            candi_vec = self.transformer.transform(candi_vec_).toarray()[0]
            dis = cosine(query_vec, candi_vec)
            # print(text, sent, dis)
            if dis < self.fuzzy_theshold_tfidf:
                score = 100
                result.append((text, 100))
                break

        for sent in self.forbidden_sents:
            candi_text = ' '.join(jieba.lcut(sent))
            candi_vec = self.transformer.transform(self.vectorizer.transform([candi_text])).toarray()[0]
            dis = cosine(query_vec, candi_vec)
            # print(text, sent, dis)
            if dis < self.fuzzy_theshold_tfidf:
                if score == 100:
                    raise Exception('同时符合文明用语和服务禁语！')
                score = 0
                result.append((text, 0))
                break
        return score, result

    def fuzzy_match_bert(self, text):
        query_vec = self.seq2vec.seq2vec(text, self.device)
        score = self.init_score
        sim_sents = []
        result = []
        for i in range(len(self.polite_sent_vecs)):
            sent = self.polite_sents[i]
            candi_vec = self.polite_sent_vecs[i]
            dis = cosine(query_vec, candi_vec)
            if dis < self.fuzzy_theshold_bert:
                sim_sents.append((sent, dis))
        if sim_sents:
            sim_sents.sort(key=lambda x: x[1])
            score = 100
            result.append((sim_sents[0][0], 100, sim_sents[0][1]))
            return score, result

        for i in range(len(self.forbidden_sents)):
            sent = self.forbidden_sents[i]
            candi_vec = self.forbidden_sent_vecs[i]
            dis = cosine(query_vec, candi_vec)
            if dis < self.fuzzy_theshold_bert:
                sim_sents.append((sent, dis))
        if sim_sents:
            sim_sents.sort(key=lambda x: x[1])
            score = 0
            result.append((sim_sents[0][0], 0, sim_sents[0][1]))
            return score, result
        return score, result

    def ppl2score(self, ppl):
        '''
        ppl         score
        0-100       100-90
        100-300     90-80
        300-700     80-70
        700-1500    70-60
        1500-7500   60-0
        >7500       0
        '''
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

    def pkuseg_cut(self, text):
        sent = re.sub('，|。|！|\!|\.|？|\?', " ", text)
        words = self.seg2.cut(sent)
        words_list = ' '.join([word for word in words if word != " "])
        return words_list

    def fluency(self, text):
        scores = []
        results = []
        texts = re.split('[。！？]', text)
        for line in texts:
            line = line.strip()
            if line == '':
                continue
            sentence = self.pkuseg_cut(line)
            # print(sentence)
            logprob = self.lang_model.score(sentence)
            words = sentence.split(' ')
            ppl = pow(10, -1.0 / (1 + len(words)) * (logprob))
            pplscore = self.ppl2score(ppl)
            # pplscores.append(pplscore)

            words = ['<s>'] + sentence.split() + ['</s>']
            for i, (prob, length, oov) in enumerate(self.lang_model.full_scores(sentence)):
                if oov:
                    print('\t"{0}" is an OOV'.format(words[i + 1]))
            idx_errors = []

            # TODO: 改错模型未实现
            # idx_errors = self.corrector.detect(''.join(sentence.split()))
            # details.append(idx_errors)
            scores.append(pplscore)
            results.append((line, pplscore, idx_errors))
        return int(np.mean(scores)), results

    def keywords(self, text, keywords):
        results = []
        texts = re.split('[。！？]', text)
        hit_words = set()
        for text in texts:
            text = text.strip()
            if text == '':
                continue
            text_tokens = jieba.lcut(text)
            hit_words_ = {token for token in text_tokens if token in keywords}
            hit_words |= hit_words_
            results.append((text, -1, list(hit_words_)))
        score = int((len(hit_words) / len(keywords)) * 100)
        return score, results

    def politeness_prohibition(self, text):
        scores = []
        results = []
        texts = re.split('[。！？]', text)
        for text in texts:
            text = text.strip()
            if text == '':
                continue
            if len(text) > 4:
                if self.embedding_type == 'leven':
                    score, result = self.fuzzy_match_Levenshtein(text)
                elif self.embedding_type == 'tfidf':
                    score, result = self.fuzzy_match_tfidf(text)
                elif self.embedding_type == 'bert':
                    score, result = self.fuzzy_match_bert(text)
                else:
                    raise NotImplementedError

                if score == self.init_score:
                    score, result = self.precision_match(text)
            else:
                score, result = self.precision_match(text)
            results.append((text, score, result))
            scores.append(score)
        return int(np.mean(scores)), results

    def dis2score(self, dis):
        '''
        dis         score
        0-0.16      100-60
        0.16-0.25      60-50
        0.25-0.4      50-40
        0.4-1      50-0
        '''
        if dis <= 0.16:
            return int((0.16 - dis) / (0.16/40) + 60)
        elif 0.16 < dis <= 0.25:
            return int((0.25 - dis) / (0.09/10) + 50)
        elif 0.25 < dis <= 0.4:
            return int((0.4 - dis) / (0.15/10) + 40)
        else:
            return 0

    def sent_sim(self, text, standards):
        standards = [s.strip() for s in standards if s.strip()!='']
        candi_vecs = []
        for standard in standards:
            candi_vecs.append(self.seq2vec.seq2vec(standard, self.device))
        candi_vecs = np.array(candi_vecs)

        scores = []
        results = []
        texts = re.split('[。！？]', text)
        for text in texts:
            text = text.strip()
            if text == '':
                continue
            query_vec = self.seq2vec.seq2vec(text, self.device)
            diss = []
            for i in range(len(candi_vecs)):
                candi_vec = candi_vecs[i]
                candi_sent = standards[i]
                dis = cosine(query_vec, candi_vec)
                diss.append((candi_sent, dis, self.dis2score(dis)))
            diss.sort(key=lambda x: x[1])
            results.append((text, self.dis2score(diss[0][1]), diss))
            scores.append(self.dis2score(diss[0][1]))

        return int(np.mean(scores)), results

    def content_completeness(self, text, standards):
        sim_score, sim_results = self.sent_sim(text, standards)
        results = []
        hit_sents = set()
        for text, _, diss in sim_results:
            match_sents = []
            match_sents_scores = []
            for i in range(len(diss)):
                dis = diss[i][1]
                sent = diss[i][0]

                if dis < self.fuzzy_theshold_bert:
                    match_sents_scores.append(self.dis2score(dis))
                    match_sents.append(sent)
                    hit_sents.add(sent)
                else:
                    break

            results.append((text, -1, match_sents))
        score = int((len(hit_sents) / len(standards)) * 100)
        return score, results, sim_score, sim_results