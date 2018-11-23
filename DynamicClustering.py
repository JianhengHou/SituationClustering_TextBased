import numpy as np
import re
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
from gensim.models import KeyedVectors
from numpy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer


class DynamicClustering(object):
    def __init__(self, output_path):
        try:
            self.f = open(output_path, 'wb')
        except IOError:
            print("Error: Please enter the right format of output file.")
        else:
            self.valid_cluster_set = {}
            self.vanishing_cluster_set = {}
            self.invaid_cluster_set = {}
            self.outlier_cluster_set = {'size': 0, 'docs': {}}

            self.id_text = {}
            # cold start threshold
            self.cold_start_threshold = 2000
            # FastText or Word2Vec threshold
            self.d_threshold = 0.9
            # TF-IDF threshold
            # self.d_threshold = 0.2
            self.N_threshold = 2000
            self.t_threshold = 60 * 60 * 24 * 1 # valid period: 1 day
            self.m_threshold = 1.3


            self.best_cluster_labels = None
            self.best_cluster_centroids = None
            self.insertion_size_vanishing_cluster = {}

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
        text = re.sub("\n+", " ", text)
        text = re.sub('"', "", text)
        pattern = re.compile(r'@\w+')
        text = re.sub(pattern, '', text)
        pattern = re.compile(r'[a-zA-z]+://[^\s]*')
        text = re.sub(pattern, '', text)
        text = re.sub(":", "", text)
        text = re.sub("-", " ", text)
        text = re.sub("#", " ", text)
        text = text.lstrip('rt ')
        text = text.lstrip('RT ')
        pattern = re.compile(r'[^a-zA-z0-9 -]')
        text = re.sub(pattern, '', text)
        text = text.strip()
        return text

    @staticmethod
    def recluster(X):
        silhouette_int = -1  # initiate Silhouette threshold
        cluster_labels_k = None
        cluster_centroids = None
        for n_clusters in range(2, 20):  # Go through from 2 to 20
            model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels_tmp = model_kmeans.fit_predict(X)  # Traning
            silhouette_tmp = silhouette_score(X, cluster_labels_tmp)  # Getting each Silhouette score for each K
            cluster_centroid = model_kmeans.cluster_centers_
            if silhouette_tmp > silhouette_int:  # if higher Silhouette score
                best_k = n_clusters  # save best k
                silhouette_int = silhouette_tmp  # save best silhouette score
                cluster_labels_k = cluster_labels_tmp  # save best label set
                cluster_centroids = cluster_centroid # save best centroids
        return cluster_labels_k, cluster_centroids

    def update_kb(self, dic=0):
        """
        Updating the clustering data structure
        :param dic:
        :return: updated dic
        """
        self.checkpoint += 1
        print('===Check point: {0} ===='.format(self.checkpoint))

        # re-cluster
        if dic == 0:
            time = datetime.now()
            self.best_cluster_labels, self.best_cluster_centroids = self.recluster(
                self.outlier_cluster_set['docs'].values())

            # cluster : [doc_id]
            id_in_cluster = {}
            for i, each in enumerate(self.best_cluster_labels):
                if each not in id_in_cluster.keys():
                    id_in_cluster[each] = []
                    id_in_cluster[each].append(self.outlier_cluster_set['docs'].keys()[i])
                else:
                    id_in_cluster[each].append(self.outlier_cluster_set['docs'].keys()[i])

            # transfer re-cluster result into the valid cluster set
            for key, value in id_in_cluster.items():
                new_key = 'cluster' + str(self.cluster_number)
                self.cluster_number += 1
                self.valid_cluster_set[new_key] = {}
                self.valid_cluster_set[new_key]['last_update_time'] = time
                self.valid_cluster_set[new_key]['size'] = len(value)
                self.valid_cluster_set[new_key]['centroid'] = self.best_cluster_centroids[key]
                self.valid_cluster_set[new_key]['centroid_norm'] = linalg.norm(self.valid_cluster_set[new_key]['centroid'])
                self.valid_cluster_set[new_key]['sum_vector'] = self.valid_cluster_set[new_key]['size'] * self.best_cluster_centroids[key]
                self.valid_cluster_set[new_key]['docs'] = value
            self.outlier_cluster_set = {'size': 0, 'docs': {}}
        else:
            self.id_text[dic['id']] = dic[self.text_original]

            # This time format is for nepal dataset
            time = datetime.strptime(dic[self.tmstp].encode('utf-8'), '%Y-%m-%dT%H:%M:%SZ')
            # #This time format for tweets data called by Twitter API
            # time = datetime.strptime(dic[self.tmstp].encode('utf-8'), '%a %b %d %H:%M:%S +0000 %Y')
            t_content = self.remove(dic[self.text_original])  # pre-processing tweets

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

            judgment_d = True
            for key, item in self.valid_cluster_set.items():
                similarity = np.dot(text_vector, item['centroid']) / (text_vector_norm * item['centroid_norm'])
                if similarity > self.d_threshold:
                    self.valid_cluster_set[key]['last_update_time'] = time
                    self.valid_cluster_set[key]['size'] += 1
                    self.valid_cluster_set[key]['sum_vector'] += text_vector
                    self.valid_cluster_set[key]['docs'].append(dic['id'])
                    self.valid_cluster_set[key]['centroid'] = 1.0 / self.valid_cluster_set[key]['size'] * self.valid_cluster_set[key]['sum_vector']
                    self.valid_cluster_set[key]['centroid_norm'] = linalg.norm(self.valid_cluster_set[key]['centroid'])
                    judgment_d = False
            if judgment_d:
                for key, item in self.vanishing_cluster_set.items():
                    similarity = np.dot(text_vector, item['centroid']) / (text_vector_norm * item['centroid_norm'])
                    if similarity > self.d_threshold:
                        self.vanishing_cluster_set[key]['last_update_time'] = time
                        self.vanishing_cluster_set[key]['size'] += 1
                        self.vanishing_cluster_set[key]['sum_vector'] += text_vector
                        self.vanishing_cluster_set[key]['docs'].append(dic['id'])
                        self.vanishing_cluster_set[key]['centroid'] = 1.0 / self.vanishing_cluster_set[key]['size'] * self.vanishing_cluster_set[key]['sum_vector']
                        self.vanishing_cluster_set[key]['centroid_norm'] = linalg.norm(self.vanishing_cluster_set[key]['centroid'])
                        judgment_d = False
            if judgment_d:
                if self.checkpoint <= self.cold_start_threshold:
                    new_key = 'cluster' + str(self.cluster_number)
                    self.cluster_number += 1
                    self.valid_cluster_set[new_key] = {}
                    self.valid_cluster_set[new_key]['last_update_time'] = time
                    self.valid_cluster_set[new_key]['size'] = 1
                    self.valid_cluster_set[new_key]['centroid'] = text_vector
                    self.valid_cluster_set[new_key]['sum_vector'] = text_vector
                    self.valid_cluster_set[new_key]['centroid_norm'] = text_vector_norm
                    self.valid_cluster_set[new_key]['docs'] = [dic['id']]
                else:
                    self.outlier_cluster_set['size'] += 1
                    self.outlier_cluster_set['docs'][dic['id']] = text_vector
            if self.checkpoint == self.cold_start_threshold:
                for key, item in self.valid_cluster_set.items():
                    if self.valid_cluster_set[key]['size'] == 1:
                        self.outlier_cluster_set['size'] += 1
                        self.outlier_cluster_set['docs'][self.valid_cluster_set[key]['docs'][0]] = \
                            self.valid_cluster_set[key]['centroid']
                        del self.valid_cluster_set[key]
            # re-cluster
            if self.outlier_cluster_set['size'] >= self.N_threshold:
                self.best_cluster_labels,self.best_cluster_centroids = self.recluster(self.outlier_cluster_set['docs'].values())

                # cluster : [doc_id]
                id_in_cluster = {}
                for i, each in enumerate(self.best_cluster_labels):
                    if each not in id_in_cluster.keys():
                        id_in_cluster[each] = []
                        id_in_cluster[each].append(self.outlier_cluster_set['docs'].keys()[i])
                    else:
                        id_in_cluster[each].append(self.outlier_cluster_set['docs'].keys()[i])

                # transfer re-cluster result into the valid cluster set
                for key, value in id_in_cluster.items():
                    new_key = 'cluster'+str(self.cluster_number)
                    self.cluster_number += 1
                    self.valid_cluster_set[new_key] = {}
                    self.valid_cluster_set[new_key]['last_update_time'] = time
                    self.valid_cluster_set[new_key]['size'] = len(value)
                    self.valid_cluster_set[new_key]['centroid'] = self.best_cluster_centroids[key]
                    self.valid_cluster_set[new_key]['centroid_norm'] =linalg.norm(self.valid_cluster_set[new_key]['centroid'])
                    self.valid_cluster_set[new_key]['sum_vector'] = self.valid_cluster_set[new_key]['size']*self.best_cluster_centroids[key]
                    self.valid_cluster_set[new_key]['docs'] = value
                self.outlier_cluster_set = {'size': 0, 'docs': {}}

            # check if there are clusters out of date  in the valid cluster set
            for key, item in self.valid_cluster_set.items():
                if (time - self.valid_cluster_set[key]['last_update_time']).seconds > self.t_threshold:
                    self.vanishing_cluster_set[key] = item
                    self.insertion_size_vanishing_cluster[key] = self.valid_cluster_set[key]['size']
                    del self.valid_cluster_set[key]
            # check if there are clusters out of date in the vanishing cluster set
            for key, item in self.vanishing_cluster_set.items():
                if (time - self.vanishing_cluster_set[key]['last_update_time']).seconds > 2 * self.t_threshold:
                    self.invaid_cluster_set[key] = item
                    del self.vanishing_cluster_set[key]
            # Check if there are clusters getting bigger enough in the vanishing cluster set
            for key, item in self.vanishing_cluster_set.items():
                if (float(self.vanishing_cluster_set[key]['size']) / self.insertion_size_vanishing_cluster[key]) > self.m_threshold:
                    self.vanishing_cluster_set[key]['last_update_time'] = time
                    self.valid_cluster_set[key] = item
                    del self.vanishing_cluster_set[key]
                    del self.insertion_size_vanishing_cluster[key]

    def kb_to_situation_clustering(self):
        """
           #     Transferring global dictionary(i.e clustering data structure) to a json-line file
           #     :output: .txt file
           #     """
        num = 1
        cluster_set = [self.valid_cluster_set, self.vanishing_cluster_set,self.invaid_cluster_set]
        for each_set in cluster_set:
            for k, v in each_set.items():
                DC_result = {}
                DC_result['cluster' + str(num)] = v['docs']
                DC_result['preferred_name'] = 'To be defined'
                self.f.write(json.dumps(DC_result)+'\n')
                num += 1
        for tweet in self.outlier_cluster_set['docs']:
            DC_result = {}
            DC_result['cluster' + str(num)] = tweet
            DC_result['preferred_name'] = 'To be defined'
            self.f.write(json.dumps(DC_result) + '\n')
            num += 1