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

product_id = []
session_max_length = 12
vocab = {}



def load_data(data_path):
    f = open(data_path, mode='r', encoding='utf-8')
    wv_model = KeyedVectors.load_word2vec_format('wv.model', binary=False)
    product_id.extend(wv_model.index2word)
    dict_size = len(product_id)
    X_sequence_product = []
    Y_target_product = []
    for line in f:
        item_in_session = line.split('\t')
        x_sequence_product = []
        for i in range(len(item_in_session) - 1):
            item = item_in_session[i].replace('\n', '')
            vector_item = wv_model.wv[item]
            x_sequence_product.append(vector_item)
        vector_target = product_id.index(item_in_session[len(item_in_session) - 1].replace('\n', ''))
        X_sequence_product.append(x_sequence_product)
        Y_target_product.append(vector_target)
    X_train = pad_sequences(X_sequence_product, session_max_length, padding='post', dtype='float32')
    Y_train = to_categorical(Y_target_product, num_classes=len(product_id))
    #print(Y_train.shape[1])
    print(len(wv_model.vocab))
    vocab = wv_model.vocab
    return X_train, Y_train

def top_k_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=10)

def train_model(data_path):
    X, Y = load_data(data_path)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)

    model = Sequential()
    model.add(InputLayer(input_shape=(12, 100)))
    model.add(Bidirectional(GRU(128, recurrent_dropout=0.35, return_sequences=True)))
    model.add(Bidirectional(GRU(128, recurrent_dropout=0.35, return_sequences=False)))
    model.add(Dense(len(product_id), activation='softmax'))
    model.summary()
    print(product_id)
    board = keras.callbacks.TensorBoard(log_dir='E:\Logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None)
    model_name = '../rs.h5'
    checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_acc', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', top_k_accuracy])
    #model.evaluate()
    history = model.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], batch_size=512, epochs=30, callbacks=[checkpointer, board])
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    train_model(data_path='../data_sample/new_train_sequence.dvg')
    #load_data()
