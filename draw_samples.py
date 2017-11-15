import random
import numpy as np
import json

from process_data import save_pickle, load_pickle, load_task

# train_data = load_task('./dataset/train-v1.1.json')
# dev_data = load_task('./dataset/dev-v1.1.json')
# save_pickle(train_data, 'pickle/train_data.pickle')
# save_pickle(dev_data, 'pickle/dev_data.pickle')

train_data = load_pickle('pickle/train_data.pickle')
dev_data = load_pickle('pickle/dev_data.pickle')

def draw_sample(data, n_samples=5, random_choice=False):
    n_data = len(data)
    for i in range(n_samples):
        print('------------------------------------------')
        # ret_data.append((c, qa['id'], q, a))
        if random_choice:
            target_id = min(n_data-1, random.randint(0, n_data-1))
        else:
            target_id = i
        target = data[target_id]
        print('Context:', ' '.join(target[0]))
        print('Question:', ' '.join(target[2]))
        print('Answer  :', ' '.join(target[3]))

draw_sample(train_data+dev_data)

print('end')