import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, Dense, Activation, InputLayer, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import sys

class Predictor:
    vocab = {}
    product_id = []
    session_max_length = 12
    project_id_map = {}
    street_id_map = {}
    ward_id_map = {}
    district_id_map = {}
    model = None
    wv_model = None

    def load_data(self):
        self.wv_model = KeyedVectors.load_word2vec_format('wv.model', binary=False)
        self.product_id = self.wv_model.index2word

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(12, 100)))
        self.model.add(Bidirectional(GRU(128, recurrent_dropout=0.35, return_sequences=True)))
        self.model.add(Bidirectional(GRU(128, recurrent_dropout=0.35, return_sequences=False)))
        self.model.add(Dense(len(self.product_id), activation='softmax'))
        self.model.load_weights('rs.h5')

    def load_static_file(self):
        f = open('db/districts.csv', mode='r', encoding='utf-8')
        for line in f:
            line = line.replace(u'\ufeff', '').replace('\n', '')
            split = line.split(',')
            self.district_id_map[split[0]] = split[1]
        f.close()
        f = open('db/projects.csv', mode='r', encoding='utf-8')
        for line in f:
            line = line.replace(u'\ufeff', '').replace('\n', '')
            split = line.split(',')
            self.project_id_map[split[0]] = split[1]
        f.close()
        f = open('db/streets.csv', mode='r', encoding='utf-8')
        for line in f:
            line = line.replace(u'\ufeff', '').replace('\n', '')
            split = line.split(',')
            self.street_id_map[split[0]] = split[1]
        f.close()
        f = open('db/wards.csv', mode='r', encoding='utf-8')
        for line in f:
            line = line.replace(u'\ufeff', '').replace('\n', '')
            split = line.split(',')
            self.ward_id_map[split[0]] = split[1]
        f.close()

    def __init__(self):
        self.load_static_file()
        self.load_data()

    def map_id_to_object(self, id):
        split = id.split('_')
        if split[0] is 'd':
            return self.district_id_map[split[1]]
        if split[0] is 'p':
            return self.project_id_map[split[1]]
        if split[0] is 's':
            return self.street_id_map[split[1]]
        if split[0] is 'w':
            return self.ward_id_map[split[1]]
        return '0'

    def predict(self, sequence_input):
        inputs_list = []
        for item in sequence_input:
            vec_item = self.wv_model.wv[item]
            inputs_list.append(vec_item)
            print(self.map_id_to_object(item))
        input = pad_sequences([inputs_list], self.session_max_length, padding='post', dtype='float32')

        output = None
        try:
            output = self.model.predict_classes(input)
        except:
            print("Unexpected error:", sys.exc_info()[0])
        id = self.product_id[int(output)]
        return id, self.map_id_to_object(id)


if __name__ == '__main__':
    predictor = Predictor()
    predictor.load_static_file()
    wv_model = KeyedVectors.load_word2vec_format('wv.model', binary=False)
    product_id = wv_model.index2word
    predictor.load_data()
    sequence = ['s_361', 's_11233', 'p_2701', 's_2070', 'd_61']
    input_list = []
    for item in sequence:
        vector_item = wv_model.wv[item]
        input_list.append(vector_item)
        print(predictor.map_id_to_object(item))
    input = pad_sequences([input_list], predictor.session_max_length, padding='post', dtype='float32')
    output = predictor.model.predict_classes(input)
    id = product_id[int(output)]
    print(predictor.map_id_to_object(id))
    print(type(output))
