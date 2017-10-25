import numpy as np
import json
import pprint

from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, LSTM, Lambda, Permute, Dropout, add, multiply, dot
from keras.layers.normalization import BatchNormalization

from process_data import save_pickle, load_pickle, load_task, vectroize
from net.simple_embedding import SimpleEmbedding

train_X = load_task('./dataset/train-v1.1.json')
save_pickle(train_X, 'pickle/train_X.pickle')

vocab = set()
for context, q, answer in train_X:
    vocab |= set(context + q + answer)
vocab = list(sorted(vocab))
w2i = dict((c, i) for i, c in enumerate(vocab, 0))
i2w = dict((i, c) for i, c in enumerate(vocab, 0))
save_pickle(vocab, 'pickle/vocab.pickle')
save_pickle(w2i, 'pickle/w2i.pickle')
save_pickle(i2w, 'pickle/i2w.pickle')
train_X = load_pickle('pickle/train_X.pickle')
vocab = load_pickle('pickle/vocab.pickle')
w2i = load_pickle('pickle/w2i.pickle')

vocab_size = len(vocab)
embd_size = 128
context_maxlen = max(map(len, (c for c, _, _ in train_X)))
question_maxlen = max(map(len, (q for _, q, _ in train_X)))
print('vocab size:', vocab_size)
print('embd size:', embd_size)
print('context_maxlen:', context_maxlen)
print('question_maxlen:', question_maxlen)


C, Q, A = vectroize(train_X, w2i, context_maxlen, question_maxlen)
model = SimpleEmbedding(question_maxlen, vocab_size, embd_size)
print(model.summary())
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(Q, A, batch_size=64, epochs=10, validation_split=.1, verbose=1)