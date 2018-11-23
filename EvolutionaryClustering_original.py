import numpy as np
import re
import json
from numpy import linalg
from gensim.models import KeyedVectors
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer


class EvolutionaryClustering_original(object):
    def __init__(self, output_path):
        try:
            self.f = open(output_path, 'wb')
        except IOError:
            print("Error: Please enter the right format of output file.")
        else:
            self.KB = {}

            self.frozen_set = {}
            self.valid_size = 2
            # FastText or Word2Vec threshold
            self.similarity_threshold = 0.9
            # #tf-idf threshold
            # self.similarity_threshold = 0.2
            self.time_threshold = 60 * 60 * 24 * 1 # valid period: 1 day
            self.frozen_threshold = 86400

            self.id_text = {}
            self.cluster_number = 1
            self.checkpoint = 0

            # Word2vec Initiation
            print('Initializing word embedding...')
            self.embedding_model = KeyedVectors.load_word2vec_format('crisisNLP_word_vector.bin', binary=True)
            self.index2word_set = set(self.embedding_model.index2word)

            self.text_original = 'text_original'
            self.tmstp = 'tmstp'

    @staticmethod
    def remove(text):
        text = text.lower()
        text = re.sub("\n+", " ", text)  # line break
        text = re.sub("#", "", text)
        text = re.sub('"', "", text)
        pattern = re.compile(r'@\w+')
        text = re.sub(pattern, '', text)
        pattern = re.compile(r'[a-zA-z]+://[^\s]*')
        text = re.sub(pattern, '', text)
        text = re.sub(":", "", text)  # colon mark
        text = re.sub("-", " ", text)  #
        text = text.lstrip('rt ')
        text = text.lstrip('RT ')
        pattern = re.compile(r'[^a-zA-z0-9 -]')
        text = re.sub(pattern, '', text)
        text = text.strip()
        return text

    def update_kb(self, dic):
        """
        Updating the clustering data structure
        :param dic:
        :return: updated dic
        """
        self.checkpoint += 1
        print('===Check point: {0} ===='.format(self.checkpoint))
        self.id_text[dic['id']] = dic[self.text_original]
        # This time format is for nepal dataset
        time = datetime.strptime(dic[self.tmstp].encode('utf-8'), '%Y-%m-%dT%H:%M:%SZ')
        # #This time format for tweets data called by Twitter API
        # time = datetime.strptime(dic[self.tmstp].encode('utf-8'), '%a %b %d %H:%M:%S +0000 %Y')
        t_content = self.remove(dic[self.text_original])

        # Word2vec sentence vector
        senvec = np.zeros(300)
        sentencesplit = t_content.split()
        word_vaild = 0
        for word in sentencesplit:
            if set([word]).intersection(self.index2word_set):
                senvec += self.embedding_model[word]
                word_vaild += 1
        if word_vaild != 0:
            text_vector = (1.0 / word_vaild) * senvec
        else:
            text_vector = senvec
        text_vector_norm = linalg.norm(text_vector)

        if self.KB == {}:
            cluster_key = 'cluster' + str(self.cluster_number)
            self.KB[cluster_key] = {'size': 1, 'last_update_time': time, 'centroid_norm': text_vector_norm,
                                    'centroid': text_vector, 'sum_vector': text_vector, 'docs': [dic['id']]}
            self.cluster_number += 1
        else:
            check_not_add = True
            best_fit_similarity = 0
            best_fit_key = None
            for key, item in self.KB.items():
                duration = (time - item['last_update_time'])
                duration_sec = duration.days * 86400 + duration.seconds
                if duration_sec > self.frozen_threshold and item['size'] < self.valid_size:
                    self.frozen_set[key] = item
                    del self.KB[key]
                else:
                    similarity = np.dot(text_vector, item['centroid']) / (text_vector_norm * item['centroid_norm'])
                    if similarity > self.similarity_threshold and duration_sec < self.time_threshold:
                        if similarity > best_fit_similarity:
                            best_fit_similarity = similarity
                            best_fit_key = key
                            check_not_add = False
            if check_not_add == False:
                self.KB[best_fit_key]['last_update_time'] = time
                self.KB[best_fit_key]['size'] += 1
                self.KB[best_fit_key]['sum_vector'] += text_vector
                self.KB[best_fit_key]['docs'].append(dic['id'])
                self.KB[best_fit_key]['centroid'] = (1.0 / self.KB[best_fit_key]['size']) * self.KB[best_fit_key]['sum_vector']
                self.KB[best_fit_key]['centroid_norm'] = linalg.norm(self.KB[best_fit_key]['centroid'])
            else:
                cluster_key = 'cluster' + str(self.cluster_number)
                self.KB[cluster_key] = {'size': 1, 'last_update_time': time, 'centroid_norm': text_vector_norm,
                                        'centroid': text_vector,'sum_vector': text_vector, 'docs': [dic['id']]}
                self.cluster_number += 1

    def kb_to_situation_clustering(self):
        """
        Transferring global dictionary(i.e clustering data structure) to a json-line file
        :output: .txt file
        """
        num = 1
        cluster_set = [self.KB, self.frozen_set]
        for each_set in cluster_set:
            for k, v in each_set.items():
                EvoluCO = {}
                EvoluCO['cluster' + str(num)] = v['docs']
                EvoluCO['preferred_name'] = 'To be defined'
                self.f.write(json.dumps(EvoluCO) + '\n')
                num +=1