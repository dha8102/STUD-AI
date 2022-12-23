import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def score_to_list(score):
    new_score = []
    this_score = score[1:-1].split(',')
    for q in this_score:
        new_score.append(float(q))
    return new_score


class Baselinedataset(Dataset):
    def __init__(self, file_path, tokenizer, modelname):
        df = pd.read_csv(file_path, encoding='cp949', low_memory=False)
        df = df.dropna()
        self.label = torch.LongTensor(df['label'].values)
        drop_rows = ['학생구분번호', '강의평가점수_str', '강의평가', 'dem', 'beh', 'label']
        scores = df['강의평가'].values.tolist()
        dem = df['dem'].values.tolist()
        beh = df['beh'].values.tolist()
        df.drop(columns=drop_rows, inplace=True)
        self.x = torch.FloatTensor(df.values)
        if modelname == 'BERT_MLP':
            self.input_ids = []
            self.att_mask = []
            for score in scores:
                encoded = tokenizer.encode_plus(score,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)
                self.input_ids.append(encoded['input_ids'])
                self.att_mask.append(encoded['attention_mask'])

        elif modelname == 'BERT_only':
            self.input_ids = []
            self.att_mask = []
            for i, score in enumerate(scores):
                text = dem[i] + beh[i] + score
                encoded = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)
                self.input_ids.append(encoded['input_ids'])
                self.att_mask.append(encoded['attention_mask'])

        elif modelname == 'dem':
            self.input_ids = []
            self.att_mask = []
            for i, score in enumerate(scores):
                text = dem[i]
                encoded = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)
                self.input_ids.append(encoded['input_ids'])
                self.att_mask.append(encoded['attention_mask'])
        elif modelname == 'beh':
            self.input_ids = []
            self.att_mask = []
            for i, score in enumerate(scores):
                text = beh[i]+' '+score
                encoded = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)
                self.input_ids.append(encoded['input_ids'])
                self.att_mask.append(encoded['attention_mask'])

        else:
            self.input_ids = torch.zeros_like(self.x)
            self.att_mask = torch.zeros_like(self.x)

        if modelname != 'MLP':
            self.input_ids = torch.LongTensor(self.input_ids)
            self.att_mask = torch.LongTensor(self.att_mask)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx], self.input_ids[idx], self.att_mask[idx]

    def class_counts(self):
        label = np.array(self.label)
        class_0 = len(label[label == 0])
        class_1 = len(label[label == 1])
        return [class_0, class_1]


def collator(items):
    return items
