# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        df = pd.read_csv(fname, encoding='utf-8')

        all_data = []
        for i in range(len(df)):
            line = df.iloc[i]
            title = str(line['title']).strip()
            text = str(line['text']).strip()
            entities = str(line['entity']).split(';')
            key_entities = str(line['key_entity']).split(';')
            neg = int(line['negative'])

            for e in entities:
                content = title + text
                index = content.find(e)
                if index + (tokenizer.max_seq_len // 2) < len(content):
                    content = content[:index + (tokenizer.max_seq_len // 2)]
                if len(content) > tokenizer.max_seq_len:
                    content = content[-(tokenizer.max_seq_len - len(e) - 3):]

                text_raw_indices = tokenizer.text_to_sequence(content)
                aspect_indices = tokenizer.text_to_sequence(e)
                aspect_len = np.sum(aspect_indices != 0)
                text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + content + ' [SEP] ' + e + ' [SEP]')
                bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
                polarity = 0
                if e in key_entities:
                    polarity = 1

                data = {
                    'text_raw_indices': text_raw_indices,
                    'aspect+indices': aspect_indices,
                    'text_bert_indices': text_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'polarity': polarity
                }
                all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
