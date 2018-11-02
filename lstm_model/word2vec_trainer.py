from gensim.models import Word2Vec


sentences = []
f = open('../data_sample/word2vec_sequence.txt', mode='r', encoding='utf-8')
for line in f:
    id_in_session = line.replace('\n', '').split('\t')
    sentences.append(id_in_session)

model = Word2Vec(min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=5)
#model.save("word2vec.model")
model.wv.save_word2vec_format('../wv.model', '../vocab.txt', binary=False, )
wv = model.wv['p_2353']
print(wv)