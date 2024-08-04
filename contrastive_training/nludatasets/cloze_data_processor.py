# -*-coding:utf-8-*-
import os
import torch
from torch.utils.data import TensorDataset, Dataset
import random
import csv


class ClozeExample(object):
    """A single training/test example for the roc dataset."""

    def __init__(self,
                 story_id,
                 context_sentence,
                 ops1,
                 ops2,
                 label=None):
        self.story_id = story_id
        self.context_sentence = context_sentence
        self.endings = [
            ops1,
            ops2,
        ]
        self.label = int(label) - 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"story_id: {self.story_id}",
            f"context_sentence: {self.context_sentence}",
            f"ending1: {self.endings[0]}",
            f"ending2: {self.endings[1]}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


def load_cloze_examples(filepath):
    _examples = []
    with open(filepath, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            if row[0] == "InputStoryid":
                continue
            story_id, s1, s2, s3, s4, ops1, ops2, label = row
            # # 4+1 mode
            # example = ClozeExample(
            #     story_id,
            #     context_sentence=' '.join((s1, s2, s3, s4)),
            #     ops1=ops1,
            #     ops2=ops2,
            #     label=label
            # )
            # 2+3 mode
            example = ClozeExample(
                story_id,
                context_sentence=' '.join((s1, s2, s3)),
                ops1=' '.join((s4, ops1)),
                ops2=' '.join((s4, ops2)),
                label=label
            )
            _examples.append(example)
    return _examples


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
        self.label = label


def convert_cloze_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    maxlen = 0
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(' '+example.context_sentence)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):

            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(' '+ending)

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
        label = example.label
        features.append(
            InputFeatures(
                example_id=example.story_id,
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


def get_cloze_dataset(data_path, split, tokenizer, max_seq_len=90, fraction=1.0):
    filename = os.path.join(data_path, 'cloze2016_{}.csv'.format(split))
    cloze_examples = load_cloze_examples(filename)

    if fraction < 1.0:
        random.shuffle(cloze_examples)
        cloze_examples = cloze_examples[:int(len(cloze_examples) * fraction)]

    copa_features = convert_cloze_examples_to_features(cloze_examples, tokenizer, max_seq_len)
    all_input_ids = torch.tensor(select_field(copa_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(copa_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(copa_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in copa_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return dataset


class ClozePairwiseDataset(Dataset):
    def __init__(self, tokenizer, data_path, split):
        super(ClozePairwiseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        filename = os.path.join(data_path, 'cloze2016_{}.csv'.format(split))
        cloze_examples = load_cloze_examples(filename)

        self.samples = []
        max_srclen, max_trglen = 0, 0
        for sample in cloze_examples:
            context_tokens = tokenizer.encode(' ' + sample.context_sentence, add_special_tokens=False)
            option_item = []
            for ending_index, ending in enumerate(sample.endings):
                context_tokens_choice = context_tokens[:]
                ending_tokens = tokenizer.encode(' ' + ending, add_special_tokens=False)
                option_item.append([context_tokens_choice, ending_tokens])
                max_srclen = max(max_srclen, len(context_tokens_choice))
                max_trglen = max(max_trglen, len(ending_tokens))
            label = sample.label
            self.samples.append((option_item, label))

        self.max_srclen = max_srclen + 2
        self.max_trglen = max_trglen + 2

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
        option_item, label = self.samples[item]
        src_input_ids, src_att_mask = [], []
        trg_input_ids, trg_att_mask = [], []
        for ctx_ids, op_ids in option_item:
            ctx_ids, ctx_att = self.convert_input_to_feature(ctx_ids, maxlen=self.max_srclen)
            end_ids, end_att = self.convert_input_to_feature(op_ids, maxlen=self.max_trglen)
            src_input_ids.append(ctx_ids)
            src_att_mask.append(ctx_att)
            trg_input_ids.append(end_ids)
            trg_att_mask.append(end_att)
        res = [src_input_ids, src_att_mask, trg_input_ids, trg_att_mask, label]
        return [torch.tensor(r) for r in res]



if __name__ == '__main__':
    path = '/home/murphy/Documents/sources/ctrltextgen/cloze_datasets'
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    examples = load_cloze_examples(os.path.join(path, 'cloze2016_test.csv'))
    print(examples[0])
    dataset = ClozePairwiseDataset(tokenizer, path, 'test')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    for batch in dataloader:
        print(batch[0].size())
        print(batch[2].size())
        print()
        break