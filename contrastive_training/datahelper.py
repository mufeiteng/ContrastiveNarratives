# -*-coding:utf-8-*-
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import multiprocessing
import json
import os
import logging

logger = logging.getLogger(__name__)

def tokenize_events(events, tokenizer):
    event2ids = dict()
    for event in events:
        tokens = tokenizer.tokenize(' ' + event)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        event2ids[event] = token_ids
    return event2ids


def multiprocessing_tokenize_data(event_list, tokenizer, implementor_func, ):
    workers = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool()
    filesize = len(event_list)
    _results = []
    for i in range(workers):
        chunk_start, chunk_end = (filesize * i) // workers, (filesize * (i + 1)) // workers
        _results.append(pool.apply_async(
            implementor_func, (event_list[chunk_start:chunk_end], tokenizer,)
        ))
    pool.close()
    pool.join()
    _total = dict()
    for _result in _results:
        samples = _result.get()
        for k in samples:
            _total[k] = samples[k]
    assert len(_total) == filesize
    return _total


class RocStoriesDataset(Dataset):
    def __init__(self, tokenizer, datapath, eventsim_path,
                 num_neg_sampling=1, max_seq_len=100, negsample_start=5, negsample_end=10):
        self.event_sim_dict = dict()
        with open(eventsim_path, 'r') as fin:
            for line in fin:
                d = json.loads(line)
                for k in d:
                    self.event_sim_dict[k] = d[k]

        stories = []
        const_stories = []
        with open(datapath, 'r') as fin:
            for line in fin:
                d = json.loads(line)

                story = [d[k] for k in ['s1','s2','s3','s4','end']]
                generated = d['generated']
                generated.sort(key=lambda x: x[1], reverse=True)
                generated = generated[:negsample_end]

                tag = False
                for e in story:
                    if e not in self.event_sim_dict:
                        tag = True
                if not tag:

                    # stories.append((story, cf_story))
                    stories.append(story)
                    const_stories.append(generated)

        self.samples = stories
        self.const_samples = const_stories
        logger.info('num of samples {}'.format(len(self.samples)))

        self.negsample_start, self.negsample_end = negsample_start, negsample_end
        self.tokenizer = tokenizer
        self.num_neg_sampling = num_neg_sampling

        self.src_evt_num = 2
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def convert_on_sample(self, events):
        event_ids = [self.tokenizer.encode(' ' + event, add_special_tokens=False) for event in events]
        source_ids = sum(event_ids[:self.src_evt_num], [])
        target_ids = sum(event_ids[self.src_evt_num:], [])
        input_ids = [self.cls_id] + source_ids + [self.sep_id] + target_ids + [self.sep_id]
        attention_mask = [1] * len(input_ids)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.pad_id)
            attention_mask.append(0)
        return input_ids, attention_mask

    def __getitem__(self, i):
        _items = self.samples[i]
        _const_candidates = self.const_samples[i]

        exp_examples = []
        exp_examples.append((_items, 1))
        for _ in range(self.num_neg_sampling):
            replace_id = random.randint(0, len(_items) - 1)  # 选位置
            candidate_neg_events = self.event_sim_dict[_items[replace_id]]
            kept = [(k, candidate_neg_events[k]) for k in candidate_neg_events]
            kept.sort(key=lambda x: x[1], reverse=True)
            start_idx = self.negsample_start
            kept = kept[start_idx:start_idx+self.negsample_end]
            neg_events, scores = zip(*kept)
            # print(neg_events, scores)
            scores = list(map(float, scores))
            neg_event = np.random.choice(
                neg_events, p=normalize(np.array(scores)))
            neg_sample = [tokens for tokens in _items]
            neg_sample[replace_id] = neg_event
            exp_examples.append((neg_sample, 0))
        random.shuffle(exp_examples)

        label = -1
        input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
        for idx, (events, label_tag) in enumerate(exp_examples):
            if label_tag == 1:
                label = idx
            input_ids, attention_mask = self.convert_on_sample(events)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            # imp_mask.append(imp_tag)
        res = [input_ids_list, attention_mask_list, label]


        exp_input_ids_list, exp_attention_mask_list, exp_label = [torch.tensor(r) for r in res]



        prefix_len = 2
        imp_examples = [(_items, 1)]
        num_sample = int(self.num_neg_sampling)

        is_replace = True if num_sample >= len(_const_candidates) else False
        kept_indices = np.random.choice(list(range(len(_const_candidates))), num_sample, replace=is_replace)

        for ind in kept_indices:
            _const_story = _const_candidates[ind][0]
            assert len(_const_story) == 5
            if random.random() < 0.5:
                neg1 = _items[:prefix_len] + _const_story[prefix_len:]
                # neg2 = _const_story[:prefix_len]+_items[prefix_len:]
                imp_examples.append((neg1, 0))
                # imp_examples.append((neg2, 0))
            else:
                neg2 = _const_story[:prefix_len + 1] + _items[prefix_len + 1:]
                # imp_examples.append((neg1, 0))
                imp_examples.append((neg2, 0))

        random.shuffle(imp_examples)
        label = -1
        input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
        for idx, (events, label_tag) in enumerate(imp_examples):
            if label_tag == 1:
                label = idx
            input_ids, attention_mask = self.convert_on_sample(events)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            # imp_mask.append(imp_tag)
        res = [input_ids_list, attention_mask_list, label]
        imp_input_ids_list, imp_attention_mask_list, imp_label = [torch.tensor(r) for r in res]
        return exp_input_ids_list, exp_attention_mask_list, exp_label, \
               imp_input_ids_list, imp_attention_mask_list, imp_label
               # item1_ids, item1_attention_mask, \
               # item2_ids, item2_attention_mask



def process_one_item(tokenizer, src, trg, maxlen):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_id = tokenizer.pad_token_id
    tokens = [cls_token] + src + [sep_token] + trg + [sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * (len(src) + 2) + [1] * (len(trg) + 1)
    if len(input_ids) > maxlen:
        input_ids = input_ids[:maxlen]
        attention_mask = attention_mask[:maxlen]
        token_type_ids = token_type_ids[:maxlen]
    while len(input_ids) < maxlen:
        input_ids.append(pad_id)
        attention_mask.append(0)
        token_type_ids.append(1)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    return input_ids, attention_mask



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize(scores, method='softmax'):
    if method == 'softmax':
        return softmax(scores)
    if method == 'average':
        return np.array(scores)/np.sum(scores)
    raise NotImplementedError('only support softmax or average')

