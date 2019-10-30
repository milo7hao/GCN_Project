

from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc



pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

DATA_PATH = Path('../data/')
LABEL_PATH = Path('../labels/')

AUG_DATA_PATH = Path('../data/data_augmentation/')

MODEL_PATH=Path('../models/')
LOG_PATH=Path('../logs/')
MODEL_PATH.mkdir(exist_ok=True)

model_state_dict = None

# BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/cased_L-12_H-768_A-12/')
BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../bert_fastai/pretrained-weights/uncased_L-24_H-1024_A-16/')
# FINETUNED_PATH = Path('../models/finetuned_model.bin')
FINETUNED_PATH = None
# model_state_dict = torch.load(FINETUNED_PATH)

LOG_PATH.mkdir(exist_ok=True)

OUTPUT_PATH = MODEL_PATH/'output'
OUTPUT_PATH.mkdir(exist_ok=True) 

label_cols = ["0", "1", "2", "3"]

#creating data bunch object
from fast_bert.data_cls import BertDataBunch

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=16, #depends
                          max_seq_length=512, #depends
                          multi_gpu=False, #depends
                          multi_label=True,
                          model_type='bert')

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

logger = logging.getLogger()
device_cuda = torch.device("cuda") #depends
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_PATH,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True, #depends
						is_fp16=True,
						multi_label=True,
						logging_steps=50)

learner.fit(epochs = 7,
lr = 6e-5,
validate= False,
optimizer_type="lamb"
)

learner.save_model()

texts = ['I really love the Netflix original movies',
		 'this movie is not worth watching'] #need to replace with our train data
predictions = learner.predict_batch(texts)
# To be used only to finetune
# args = Box({
#     "run_text": "multilabel toxic comments with freezable layers",
#     "train_size": -1,
#     "val_size": -1,
#     "log_path": LOG_PATH,
#     "full_data_dir": DATA_PATH,
#     "data_dir": DATA_PATH,
#     "task_name": "toxic_classification_lib",
#     "no_cuda": False,
#     "bert_model": BERT_PRETRAINED_PATH,
#     "output_dir": OUTPUT_PATH,
#     "max_seq_length": 512,
#     "do_train": True,
#     "do_eval": True,
#     "do_lower_case": True,
#     "train_batch_size": 8,
#     "eval_batch_size": 16,
#     "learning_rate": 5e-5,
#     "num_train_epochs": 6,
#     "warmup_proportion": 0.0,
#     "no_cuda": False,
#     "local_rank": -1,
#     "seed": 42,
#     "gradient_accumulation_steps": 1,
#     "optimize_on_cpu": False,
#     "fp16": True,
#     "fp16_opt_level": "O1",
#     "weight_decay": 0.0,
#     "adam_epsilon": 1e-8,
#     "max_grad_norm": 1.0,
#     "max_steps": -1,
#     "warmup_steps": 500,
#     "logging_steps": 50,
#     "eval_all_checkpoints": True,
#     "overwrite_output_dir": True,
#     "overwrite_cache": False,
#     "seed": 42,
#     "loss_scale": 128,
#     "task_name": 'intent',
#     "model_name": 'xlnet-base-cased',
#     "model_type": 'xlnet'
# })