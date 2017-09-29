# *Attentive* LMs

This repository contains the code for the IJCNLP'2017 paper *Attentive* Language Models. The code is largely based on [Tensorflow's tutorial on Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent). The code allow to train models using both *single* and *combined* scores as described in the paper on the PTB and [wikitext2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) datasets, although it is likely to be used with other datasets.

## Software requirements

* Python 3
* [Tensorflow](https://www.tensorflow.org/install/) >= 1.3.0
* [NLTK](http://www.nltk.org/install.html)
  * We also need to install NLTK's default sentence tokenizer in order to split the sentences in wikitext2.  

## Data setup

To download PTB and to download and pre-process the wikitext2, please run
```
./get_and_process_data.sh
```
from the root directory. This script will create a folder called `data` containing the downloaded datasets (each one in its respective folder).

The scripts for training the models expect the datasets to be in this folder.


## Running the experiments

You can control the dataset and model to train using the `--config` flag:

* `ptb_single` trains an Attentive-LM with *single* score on the PTB
* `ptb_combined` trains an Attentive-LM with *combined* score on the PTB
* `wikitext2_single` trains an Attentive-LM with *single* score on the wikitext2
* `wikitext2_combined` trains an Attentive-LM with *combined* score on the wikitext2

There are 2 options to run the experiments:

#### Running start_train.sh:

Simply open `start_train.sh` and change the  `--config` flag in the script to the model you wish to train. Then run from the root folder:

```
./start_train.sh
```

#### Calling the scripts directly in python:

```
export MODEL_DIR=${HOME}/train_lms/ptb_single
mkdir -p $MODEL_DIR
python3 -u train_attentive_lm.py \
  --config="ptb_single" \
  --train_dir=$MODEL_DIR \
  --best_models_dir=$MODEL_DIR/best_models
```
