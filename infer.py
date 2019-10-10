# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorch_transformers import BertModel

from data_utils import Tokenizer4Bert, pad_and_truncate
from models import BERT_SPC

if __name__ == '__main__':
    model_classes = {
        'bert_spc': BERT_SPC
    }
    model_state_dict_paths = {
        'bert_spc': 'state_dict/bert_spc_val_acc0.9372'
    }

    class Option(object): pass
    opt = Option()
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.max_seq_len = 512
    opt.pretrained_bert_name = './chinese_wwm_ext_pytorch/'
    opt.polarities_dim = 2
    opt.dropout = 0.1
    opt.bert_dim = 768
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = opt.model_class(bert, opt).to(opt.device)

    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(opt.state_dict_path))
    model.eval()
    torch.autograd.set_grad_enabled(False)

    df = pd.read_csv('./data/Test_Data.csv', encoding='utf-8')

    with open('./data/submission.csv', 'w', encoding='utf-8') as f:
        f.write('id,negative,key_entity\n')

    for i in tqdm(range(len(df))):
        line = df.iloc[i]
        id = str(line[0])
        title = str(line[1])
        text = str(line[2])
        entities = str(line[3]).split(';')

        key_entity = []
        for e in entities:
            content = title
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

            text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
            bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)

            inputs = [text_bert_indices, bert_segments_ids]
            outputs = model(inputs)
            t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
            sentiment = t_probs.argmax(axis=-1)
            if sentiment == 1:
                key_entity.append(e)

        # remove infered key entities that can be substring of other entities.
        final_res = []
        for e1 in key_entity:
            flag = 0
            for e2 in key_entity:
                if e1 == e2:
                    continue
                if e2.find(e1) != -1:
                    flag = 1
            if flag == 0:
                final_res.append(e1)

        with open('./data/submission.csv', 'a', encoding='utf-8') as f:
            if len(final_res) > 0:
                f.write(id + ',' + '1' + ',' + ';'.join(final_res) + '\n')
            else:
                f.write(id + ',' + '0' + ',' + '\n')
