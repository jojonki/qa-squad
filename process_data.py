import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize

def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)

def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)

def load_task(dataset_path):
    ret_data = []
    # vocab = []
    with open(dataset_path) as f:
        data = json.load(f)
        ver = data['version']
        print('dataset version:', ver)
        data = data['data']
        for i, d in enumerate(data):
            print('load', d['title'], i, '/', len(data))
            for p in d['paragraphs']:
                c = word_tokenize(p['context'])
                # vocab += c
                q, a = [], []
                for qa in p['qas']:
                    q = word_tokenize(qa['question'])
                    # vocab += q
                    a = [ans['text'] for ans in qa['answers']]
                    # for ta in qa['answers']:
                        # vocab += [ta['text']]
                        # break
                    ret_data.append((c, q, a))
    return ret_data


def vectroize(data, w2i, ctx_maxlen, qst_maxlen):
    C, Q, A = [], [], []
    for i, (context, question, answer) in enumerate(data):
        if i % 20000: print('vectroize:', i, '/', len(data))
        # not use context
#         c = [w2i[w] for w in context if w in w2i]
#         c = c[:ctx_maxlen]
#         c_pad_len = max(0, ctx_maxlen - len(c))
#         c += [0] * c_pad_len

        q = [w2i[w] for w in question if w in w2i]
        q = q[:qst_maxlen]
        q_pad_len = max(0, qst_maxlen - len(q))
        q += [0] * q_pad_len

        y = np.zeros(len(w2i))
        y[w2i[answer[0]]] = 1

#         C.append(c)
        Q.append(q)
        A.append(y)
    
#     C = np.array(C)#, dtype=np.uint32)
    Q = np.array(Q)
    A = np.array(A, dtype='byte')

    return C, Q, A