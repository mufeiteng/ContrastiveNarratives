# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import ujson as json
import cjjpy as cjj
# import tensorflow as tf
from collections import defaultdict

class MultiDataLoader:
    def __init__(self, data_type, tokenizer=None, workdir=None):
        self.data_type = data_type
        if not workdir:
            self.workdir = os.environ.get('PJ_HOME', cjj.AbsParentDir(__file__, '..'))
        else:
            self.workdir = workdir
        self.tokenizer = tokenizer

    def load_data(self, role):
        return self._load_data_timetravel(role)

    def _load_data_timetravel(self, role):
        '''
        :return: premise, initial, counterfactual, original_ending, (edited_ending)
        '''
        # assert role in ['dev_data', 'test_data', 'train_unsupervised',
        #                 'train_supervised_small', 'train_supervised_large']
        data = []
        with tf.io.gfile.GFile(os.path.join(self.workdir, f'data/TimeTravel/{role}.json')) as f:
            for line in f:
                js = json.loads(line)
                data.append(js)
        return data


class DataReader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = defaultdict(list)
        self.cnt = 0

    def tokenize(self, text):
        token_id = self.tokenizer.encode(text, add_special_tokens=False)
        bert_tokenized = self.tokenizer.decode(token_id)
        return token_id, bert_tokenized

    def load(self, samples: list):
        self.cnt = len(samples)
        for d in samples:
            self._load(d)

        for k in self.data:
            assert self.cnt == len(self.data[k]), k

    def _load(self, example):
        '''
        NOTE: ADD a blank before all texts except for premise!!!
        :param example: {k: v, k: v}
        :return: {k: [{}], 'original_ending': [[{}, {}, {}], []]}
        '''
        if isinstance(example, str):
            example = json.loads(example)
        assert isinstance(example, dict)
        for k in ['split_original_end', 'premise', 'initial',
                  'counterfactual', 'original_ending',
                  'story_id', 'edited_endings',
                  ]:

            if k == 'split_original_end':
                # assert k == 'split_original_end', k
                assert len(example[k]) == 3
                tmp = []
                for ex in example[k]:
                    token_id, bert_tokenized = self.tokenize(' ' + ex.strip())
                    tmp.append({
                        'token_ids': np.array(token_id),
                        'text': bert_tokenized
                    })
                self.data[k].append(tmp)
            else:
                if k in {'story_id', 'edited_endings', 'original_ending'}:
                    self.data[k].append(example[k])
                else:
                    text = example[k].strip() if k == 'premise' else ' ' + example[k].strip()
                    token_id, bert_tokenized = self.tokenize(text)
                    self.data[k].append({
                        'token_ids': np.array(token_id),
                        'text': bert_tokenized
                    })

    def append_data(self, x1, x2):
        x1['text'] = x1['text'] + x2['text']
        x1['token_ids'] = self.tokenizer.encode(x1['text'], add_special_tokens=False)
        return x1

    def __len__(self):
        return self.cnt


if __name__ == '__main__':
    from ctrltextgen.tools import bert_base_path, project_data_path
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_base_path)
    datareader = DataReader(tokenizer)
    dataset_dir = os.path.join(project_data_path, 'timetravel')
    filepath = os.path.join(dataset_dir, 'test_data_original_end_splitted.json')
    datareader.load(filepath)
    # print(len(datareader.data))
    print(datareader.data.keys())
    # for i in datareader.data:
    #     d = datareader.data[i]
    data = datareader.data
    # premise = data['premise']
    # split_original_end = data['split_original_end']
    # print(split_original_end)