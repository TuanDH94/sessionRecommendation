from lstm_model.predictor import Predictor
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from lstm_model.trainer import load_data
import numpy as np

# load data
predictor = Predictor()
predictor.load_static_file()
wv_model = KeyedVectors.load_word2vec_format('wv.model', binary=False)
product_id = wv_model.index2word
predictor.load_data()

sequences = []
test_data_path = 'data_sample/new_test_sequence.dvg'
f = open(test_data_path, mode='r', encoding='utf-8')
writer = open('test_result.txt', mode='w', encoding='utf-8')

for line in f:
    line = line.replace('\n', '').replace(u'\ufeff', '')
    split = line.split('\t')
    sequence = split[:-1]
    true_id_output = split[-1]
    item_true_output = predictor.map_id_to_object(true_id_output)
    sequences.append(sequences)
    input_list = []
    for id_item in sequence:
        vector_item = wv_model.wv[id_item]
        input_list.append(vector_item)
        writer.write(predictor.map_id_to_object(id_item) + '\t')
    input = pad_sequences([input_list], predictor.session_max_length, padding='post', dtype='float32')
    output = predictor.model.predict_classes(input)
    id_output = product_id[int(output)]
    item_output = predictor.map_id_to_object(id_output)
    writer.write("\tPredict: " + str(item_output) + "\tTrue: " + str(item_true_output))
    writer.write('\n')