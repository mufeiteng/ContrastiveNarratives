import csv
import json
import logging
import os
import random
from abc import ABC
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from ctrltextgen.tools import load_json_samples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        label = int(label)
        assert 0 <= label < 4
        self.label = label


def convert_hellaswag_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    maxlen = 0
    features = []
    for example_index, example in enumerate(examples):
        ctx_a = example['ctx_a']
        ctx_b = example['ctx_b']
        endings = example['endings']
        label = example['label']

        context_tokens = tokenizer.tokenize(' '+ctx_a)

        choices_features = []
        for ending_index, ending in enumerate(endings):

            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(' '+ctx_b+' '+ending)

            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = [cls_token] + context_tokens_choice + [sep_token] + ending_tokens + [sep_token]
            maxlen = max(maxlen, len(tokens))
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [pad_token_id] * (max_seq_length - len(input_ids))
            input_mask += [0] * (max_seq_length - len(input_ids))
            segment_ids += [1] * (max_seq_length - len(input_ids))
            input_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((input_ids, input_mask, segment_ids))
        features.append(
            InputFeatures(
                example_id=example['video_id'],
                choices_features=choices_features,
                label=label
            )
        )
    return features


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def read_swag_instances(filepath):
    _examples = []
    with open(filepath, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            if row[1] == "video-id":
                continue
            xid, video_id, fold_ind, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, label = row
            d = {
                'ctx_a': sent1,
                'ctx_b': sent2,
                'endings': [ending0, ending1, ending2, ending3],
                'label': label,
                'video_id': video_id,
            }
            _examples.append(d)
    return _examples


def get_swag_dataset(data_path, split, tokenizer, max_seq_len=90, fraction=1.0):
    filename = os.path.join(data_path, '{}.csv'.format(split))

    _examples = read_swag_instances(filename)

    if fraction < 1.0:
        random.shuffle(_examples)
        _examples = _examples[:int(len(_examples) * fraction)]

    copa_features = convert_hellaswag_examples_to_features(_examples, tokenizer, max_seq_len)
    all_input_ids = torch.tensor(select_field(copa_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(copa_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(copa_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in copa_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return dataset


