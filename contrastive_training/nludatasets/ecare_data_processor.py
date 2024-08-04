# -*-coding:utf-8-*-
import os
import json
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import random


class ECareExample(object):
    """A single training/test example for the roc dataset."""

    def __init__(self,
                 index,
                 premise,
                 ask_for,
                 hypothesis1,
                 hypothesis2,
                 label,
                ):
        self.index = index
        self.ask_for = ask_for
        self.premise = premise
        self.endings = [
            hypothesis1,
            hypothesis2,
        ]
        self.label = int(label)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"index: {self.index}",
            f"ask_for: {self.ask_for}",
            f"premise: {self.premise}",
            f"ending1: {self.endings[0]}",
            f"ending2: {self.endings[1]}",
            f"label: {self.label}"

        ]
        return ", ".join(l)


def read_ecare_examples(input_file):
    examples = []
    with open(input_file, 'r') as fin:
        for line in fin:
            d = json.loads(line)
            sample = ECareExample(
                index=d['index'],
                premise=d['premise'],
                ask_for=d['ask-for'],
                hypothesis1=d['hypothesis1'],
                hypothesis2=d['hypothesis2'],
                label=d['label']
            )
            examples.append(sample)
    return examples


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
                'tokens': tokens,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for tokens, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def convert_copa_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(' '+example.premise)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):

            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(' '+ending)

            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            if example.ask_for == 'cause':
                tokens = [cls_token] + ending_tokens + [sep_token] + context_tokens_choice + [sep_token]
            else:
                tokens = [cls_token] + context_tokens_choice + [sep_token] + ending_tokens + [sep_token]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [pad_token_id] * (max_seq_length - len(input_ids))
            _mask = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += _mask
            segment_ids += _mask
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))
        label = example.label
        features.append(
            InputFeatures(
                example_id=example.index,
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


def get_ecare_dataset(data_path, split, tokenizer, max_seq_len, fraction=1.0):
    filename = os.path.join(data_path, '{}.jsonl'.format(split))
    ecare_samples = read_ecare_examples(filename)
    if fraction < 1.0:
        random.shuffle(ecare_samples)
        ecare_samples = ecare_samples[:int(len(ecare_samples) * fraction)]

    ecare_features = convert_copa_examples_to_features(ecare_samples, tokenizer, max_seq_len)
    all_input_ids = torch.tensor(select_field(ecare_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(ecare_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(ecare_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in ecare_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return dataset



class EcarePairwiseDataset(Dataset):
    def __init__(self, tokenizer, data_path, split):
        super(EcarePairwiseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        filename = os.path.join(data_path, '{}.jsonl'.format(split))
        ecare_samples = read_ecare_examples(filename)
        self.samples = []
        max_srclen, max_trglen = 0, 0
        for sample in ecare_samples:
            direction = sample.ask_for
            context_tokens = tokenizer.encode(' ' + sample.premise, add_special_tokens=False)
            max_srclen = max(max_srclen, len(context_tokens))

            choices_features = []
            for ending_index, ending in enumerate(sample.endings):
                context_tokens_choice = context_tokens[:]
                ending_tokens = tokenizer.encode(' ' + ending, add_special_tokens=False)
                max_trglen = max(max_trglen, len(ending_tokens))
                choices_features.append((context_tokens_choice, ending_tokens))
            label = sample.label
            self.samples.append((choices_features, direction, label))
        # self.max_srclen = max_srclen+2
        # self.max_trglen = max_trglen+2
        self.maxlen = max(max_srclen, max_trglen)+2

    def convert_input_to_feature(self, inputs, maxlen):
        input_ids = [self.cls_id] + inputs + [self.sep_id]
        if len(input_ids) > maxlen:
            input_ids = input_ids[:maxlen]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < maxlen:
            input_ids.append(self.pad_id)
            attention_mask.append(0)
        return input_ids, attention_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        option_item, direction, label = self.samples[item]
        src_input_ids, src_att_mask = [], []
        trg_input_ids, trg_att_mask = [], []

        for ctx_tokens, end_tokens in option_item:
            ctx_ids, ctx_att = self.convert_input_to_feature(ctx_tokens, maxlen=self.maxlen)
            end_ids, end_att = self.convert_input_to_feature(end_tokens, maxlen=self.maxlen)
            if direction == 'cause':
                src_ids, src_att = ctx_ids, ctx_att
                trg_ids, trg_att = end_ids, end_att
            else:
                trg_ids, trg_att = ctx_ids, ctx_att
                src_ids, src_att = end_ids, end_att
            src_input_ids.append(src_ids)
            src_att_mask.append(src_att)
            trg_input_ids.append(trg_ids)
            trg_att_mask.append(trg_att)
        res = [src_input_ids, src_att_mask, trg_input_ids, trg_att_mask, label]
        return [torch.tensor(r) for r in res]


if __name__ == '__main__':
    path = '/home/murphy/Documents/sources/ctrltextgen/e-CARE/dataset/Causal_Reasoning'
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    dataset = EcarePairwiseDataset(tokenizer, path, 'dev')
    print(dataset.maxlen, dataset.maxlen)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    for batch in dataloader:
        print(batch[0].size())
        print(batch[2].size())
        print()
        break

