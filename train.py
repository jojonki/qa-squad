import datetime
import numpy as np
import json

from keras.callbacks import ModelCheckpoint, Callback

from process_data import save_pickle, load_pickle, load_task, vectorize
from net.simple_embedding import SimpleEmbedding

train_data = load_task('./dataset/train-v1.1.json')
dev_data = load_task('./dataset/dev-v1.1.json')
save_pickle(train_data, 'pickle/train_data.pickle')
save_pickle(dev_data, 'pickle/dev_data.pickle')

vocab = set()
for context, _, q, answer in train_data+dev_data:
    vocab |= set(context + q + answer)
vocab = list(sorted(vocab))
w2i = dict((c, i) for i, c in enumerate(vocab, 0))
i2w = dict((i, c) for i, c in enumerate(vocab, 0))
save_pickle(vocab, 'pickle/vocab.pickle')
save_pickle(w2i, 'pickle/w2i.pickle')
save_pickle(i2w, 'pickle/i2w.pickle')
train_data = load_pickle('pickle/train_data.pickle')
vocab = load_pickle('pickle/vocab.pickle')
w2i = load_pickle('pickle/w2i.pickle')

vocab_size = len(vocab)
embd_size = 128
context_maxlen = max(map(len, (c for c, _, _, _ in train_data)))
question_maxlen = max(map(len, (q for _, _, q, _ in train_data)))
print('vocab size:', vocab_size)
print('embd size:', embd_size)
print('context_maxlen:', context_maxlen)
print('question_maxlen:', question_maxlen)


C, Q, A = vectorize(train_data, w2i, context_maxlen, question_maxlen)
model = SimpleEmbedding(question_maxlen, vocab_size, embd_size)
print(model.summary())

now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_path = now + '_simple-embd-' + '-{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(Q, A, 
          batch_size=64, 
          epochs=10, 
          validation_split=.1, 
          callbacks=callback_list,
          verbose=1)
