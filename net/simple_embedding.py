from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, LSTM, Lambda, Permute, Dropout, add, multiply, dot
from keras.layers.normalization import BatchNormalization

def SimpleEmbedding(question_maxlen, vocab_size, embd_size):
    q = Input((question_maxlen,), name='Q_Input')
    embd_q = Embedding(input_dim=vocab_size, output_dim=embd_size)(q)
    embd_q = Lambda(lambda x: K.sum(x, axis=1)) (embd_q)
    y = Dense(vocab_size)(embd_q)
    y = BatchNormalization()(y)
    y = Activation('softmax')(y)
    model = Model(q, y)
    
    return model