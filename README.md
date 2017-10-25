# qa-squad
A QA System on SQuAD

## Setup

First of all, you need to download dataset.
```
$ ./download.sh
```

## Train

You can use presaved pickle files or build data dictionaries with `process_data.py`.

```
$ python train.py
```
## Generate a prediction file
```
$ python generate_predictions.py -m saved_keras_model_path -o prediction_file_name
```

## Evaluate

Use an official script to obtain evaluation result.
```
$ python evaluate-v1.1.py ./dataset/dev-v1.1.json prediction_file_name
{"exact_match": XXXX, "f1": YYYY}
```

