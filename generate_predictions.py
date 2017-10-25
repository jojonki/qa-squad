import json
import numpy as np
import argparse
from keras.models import load_model
from process_data import load_pickle, load_task, vectorize

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model',
                    required=True,
                    help='saved keras model')
parser.add_argument('-o', '--output',
                    default='predictions.txt',
                    help='saved keras model')
args = parser.parse_args()
print(args)

context_maxlen = None
question_maxlen = 60

vocab    = load_pickle('pickle/vocab.pickle')
w2i      = load_pickle('pickle/w2i.pickle')
i2w      = load_pickle('pickle/i2w.pickle')
# dev_data = load_task('./dataset/dev-v1.1.json')
dev_data = load_pickle('pickle/dev_data.pickle')

model = load_model(args.model)
C, Q, A = vectorize(dev_data, w2i, context_maxlen, question_maxlen)
predicts = model.predict(Q, batch_size=64, verbose=0)   

result = {}
for i, pred in enumerate(predicts):
    ans = i2w[np.argmax(pred)]
    result[dev_data[i][1]] = ans # key: question_id, val: entity

with open(args.output, 'w') as outfile:
    print('write prediction results to', args.output)
    json.dump(result, outfile)
    print('done!')