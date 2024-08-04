# -*-coding:utf-8-*-
import os
import pickle
import random
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import json
import numpy as np
from tqdm import tqdm


def load_timetravel_train_examples(tokenizer, data_path, cache_file):
    if os.path.exists(cache_file):
        _samples = pickle.load(open(cache_file, 'rb'))
        print('load tokenized samples from {}'.format(cache_file))
    else:
        _samples = []
        with open(data_path, 'r') as fin:
            for line in fin:
                d = json.loads(line)
                premise = d['premise']
                initial = d['initial']
                original_ending = d['original_ending']
                counterfactual = d['counterfactual']
                x_tokens = tokenizer.tokenize(' '+' '.join((premise, initial)))
                x_prime_tokens = tokenizer.tokenize(' ' + ' '.join((premise, counterfactual)))
                y_tokens = tokenizer.tokenize(' '+original_ending)
                _samples.append((x_tokens, x_prime_tokens, y_tokens))
        pickle.dump(_samples, open(cache_file, 'wb'))
        print('save tokenized samples to {}'.format(cache_file))

    max_x_len, max_y_len = 0, 0
    for x_tokens, x_prime_tokens, y_tokens in _samples:
        max_x_len = max(max_x_len, len(x_tokens), len(x_prime_tokens))
        max_y_len = max(max_y_len, len(y_tokens))
    return _samples, max_x_len+max_y_len+5


class TimeTravelUnsuperTrainDataset(Dataset):
    """
    目的是衡量判别器能否区分反事实扰动. 那么应该
    1. 考察C的时候,将原始end排在三个反事实结尾之前.
    2. 考察E的时候,将原始end排在三个反事实结尾之后.
    3. 严格匹配rank,然后算acc

    """
    def __init__(self, tokenizer, samples, max_len):
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.x_list = []
        self.x_prime_list = []
        self.y_list = []

        for x_tokens, x_prime_tokens, y_tokens in samples:
            self.x_list.append(x_tokens)
            self.x_prime_list.append(x_prime_tokens)
            self.y_list.append(y_tokens)
        self.max_len = max_len

    def __len__(self):
        return len(self.x_list)

    def process_one_sample(self, x, y, maxlen, label):
        input_tokens = [self.cls_token] + x + [self.sep_token] + y + [self.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        assert len(input_ids) <= maxlen
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(x) + 2) + [1] * (len(y) + 1)
        while len(input_ids) < maxlen:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(1)
        return input_ids, attention_mask, token_type_ids, label

    def __getitem__(self, i):
        x = self.x_list[i]
        y = self.y_list[i]
        x_prime = self.x_prime_list[i]

        x_pos_item = self.process_one_sample(x, y, self.max_len, 1)
        x_neg_item = self.process_one_sample(x_prime, y, self.max_len, 0)
        x_items = [x_neg_item, x_pos_item]
        random.shuffle(x_items)

        input_ids, attention_mask, token_type_ids, x_labels = zip(*x_items)
        label = None
        for i, l in enumerate(x_labels):
            if l == 1:
                label = i
        assert label is not None
        item = [
            input_ids, attention_mask, token_type_ids, label,
        ]
        item = [torch.tensor(r) for r in item]
        return item


class TimeTravelEvalDataset(Dataset):
    """
    目的是衡量判别器能否区分反事实扰动. 那么应该
    1. 考察C的时候,将原始end排在三个反事实结尾之前.
    2. 考察E的时候,将原始end排在三个反事实结尾之后.
    3. 严格匹配rank,然后算acc

    """
    def __init__(self, tokenizer, data_path):
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.samples = []
        self.x_list = []
        self.x_prime_list = []
        self.y_list = []
        self.y_prime_list = []

        max_len = 0
        with open(data_path, 'r') as fin:
            for line in fin:
                d = json.loads(line)
                self.samples.append(d)
                premise = d['premise']
                initial = d['initial']
                original_ending = d['original_ending']
                counterfactual = d['counterfactual']
                edited_endings = d['edited_endings']
                x_tokens = tokenizer.tokenize(' ' + ' '.join((premise, initial)))
                y_tokens = tokenizer.tokenize(' ' + original_ending)

                x_prime_tokens = tokenizer.tokenize(' ' + ' '.join((premise, counterfactual)))
                max_len = max(max_len, len(x_tokens) + len(y_tokens))
                max_len = max(max_len, len(x_prime_tokens) + len(y_tokens))

                self.x_list.append(x_tokens)
                self.x_prime_list.append(x_prime_tokens)
                self.y_list.append(y_tokens)
                y_primes_tokens = []
                for end in edited_endings:
                    y_prime_tokens = tokenizer.tokenize(' ' + ' '.join(end))
                    y_primes_tokens.append(y_prime_tokens)
                    max_len = max(max_len, len(x_prime_tokens) + len(y_prime_tokens))
                    max_len = max(max_len, len(x_tokens) + len(y_prime_tokens))
                self.y_prime_list.append(y_primes_tokens)

        self.max_len = max_len+5

    def __len__(self):
        return len(self.samples)

    def process_one_sample(self, x, y, maxlen, label):

        input_tokens = [self.cls_token] + x + [self.sep_token] + y + [self.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        assert len(input_ids) <= maxlen
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(x) + 2) + [1] * (len(y) + 1)
        while len(input_ids) < maxlen:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(1)
        return input_ids, attention_mask, token_type_ids, label

    def __getitem__(self, i):
        x = self.x_list[i]
        y = self.y_list[i]
        x_prime = self.x_prime_list[i]
        y_primes = self.y_prime_list[i]

        x_items = []
        x_pos_item = self.process_one_sample(x, y, self.max_len, 1)
        x_items.append(x_pos_item)
        for y_prime in y_primes:
            x_neg_item = self.process_one_sample(x, y_prime, self.max_len, 0)
            x_items.append(x_neg_item)
        x_input_ids, x_attention_mask, x_token_type_ids, x_labels = zip(*x_items)

        x_prime_items = []
        x_prime_neg = self.process_one_sample(x_prime, y, self.max_len, 1)
        x_prime_items.append(x_prime_neg)
        for y_prime in y_primes:
            x_prime_pos = self.process_one_sample(x_prime, y_prime, self.max_len, 0)
            x_prime_items.append(x_prime_pos)
        x_prime_input_ids, x_prime_attention_mask, x_prime_token_type_ids, x_prime_labels = zip(*x_prime_items)

        item = [
            x_input_ids, x_attention_mask, x_token_type_ids, x_labels,
            x_prime_input_ids, x_prime_attention_mask, x_prime_token_type_ids, x_prime_labels
        ]

        item = [torch.tensor(r) for r in item]
        return item




class CfPairwiseDataset(Dataset):
    """
    目的是衡量判别器能否区分反事实扰动. 那么应该
    1. 考察C的时候,将原始end排在三个反事实结尾之前.
    2. 考察E的时候,将原始end排在三个反事实结尾之后.
    3. 严格匹配rank,然后算acc

    """

    def __init__(self, tokenizer, data_path):
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.samples = []
        self.x_list = []
        self.x_prime_list = []
        self.y_list = []
        self.y_prime_list = []
        self.src_maxlen, self.trg_maxlen = 0, 0
        max_len = 0
        with open(data_path, 'r') as fin:
            for line in fin:
                d = json.loads(line)
                self.samples.append(d)
                premise = d['premise']
                initial = d['initial']
                original_ending = d['original_ending']
                counterfactual = d['counterfactual']
                edited_endings = d['edited_endings']
                x_tokens = tokenizer.tokenize(' ' + ' '.join((premise, initial)))
                y_tokens = tokenizer.tokenize(' ' + original_ending)
                x_prime_tokens = tokenizer.tokenize(' ' + ' '.join((premise, counterfactual)))

                self.x_list.append(x_tokens)
                self.x_prime_list.append(x_prime_tokens)
                self.src_maxlen = max(self.src_maxlen, len(x_tokens))
                self.src_maxlen = max(self.src_maxlen, len(x_prime_tokens))

                self.y_list.append(y_tokens)
                self.trg_maxlen = max(self.trg_maxlen, len(y_tokens))
                y_primes_tokens = []
                for end in edited_endings:
                    y_prime_tokens = tokenizer.tokenize(' ' + ' '.join(end))
                    y_primes_tokens.append(y_prime_tokens)
                    self.trg_maxlen = max(self.trg_maxlen, len(y_prime_tokens))
                self.y_prime_list.append(y_primes_tokens)

        self.src_maxlen, self.trg_maxlen = self.src_maxlen + 2, self.trg_maxlen + 2

    def __len__(self):
        return len(self.samples)

    def process_one_sample(self, x_tokens, y_tokens, label):
        src_input_ids = self.tokenizer.convert_tokens_to_ids([self.cls_token] + x_tokens + [self.sep_token])
        if len(src_input_ids) > self.src_maxlen:
            src_input_ids = src_input_ids[:self.src_maxlen]
        src_attention_mask = [1] * len(src_input_ids)
        while len(src_input_ids) < self.src_maxlen:
            src_input_ids.append(self.pad_token_id)
            src_attention_mask.append(0)

        trg_input_ids = self.tokenizer.convert_tokens_to_ids([self.cls_token] + y_tokens + [self.sep_token])
        if len(trg_input_ids) > self.trg_maxlen:
            trg_input_ids = trg_input_ids[:self.trg_maxlen]
        trg_attention_mask = [1] * len(trg_input_ids)
        while len(trg_input_ids) < self.trg_maxlen:
            trg_input_ids.append(self.pad_token_id)
            trg_attention_mask.append(0)
        item = [src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask, label]
        return item

    def __getitem__(self, i):
        x = self.x_list[i]
        y = self.y_list[i]
        x_prime = self.x_prime_list[i]
        y_primes = self.y_prime_list[i]

        x_items = []
        x_pos_item = self.process_one_sample(x, y, 1)
        x_items.append(x_pos_item)
        for y_prime in y_primes:
            x_neg_item = self.process_one_sample(x, y_prime, 0)
            x_items.append(x_neg_item)
        x_src_input_ids, x_src_attention_mask, x_trg_input_ids, x_trg_attention_mask, x_labels = zip(*x_items)

        x_prime_items = []
        x_prime_neg = self.process_one_sample(x_prime, y, 1)
        x_prime_items.append(x_prime_neg)
        for y_prime in y_primes:
            x_prime_pos = self.process_one_sample(x_prime, y_prime, 0)
            x_prime_items.append(x_prime_pos)
        # print(x_prime_items)

        xp_src_input_ids, xp_src_attention_mask, xp_trg_input_ids, xp_trg_attention_mask, xp_labels = zip(*x_prime_items)

        item = [
            x_src_input_ids, x_src_attention_mask, x_trg_input_ids, x_trg_attention_mask, x_labels,
            xp_src_input_ids, xp_src_attention_mask, xp_trg_input_ids, xp_trg_attention_mask, xp_labels
        ]

        item = [torch.tensor(r) for r in item]
        return item
