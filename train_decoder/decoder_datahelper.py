# -*-coding:utf-8-*-
import os
import random
import json
import torch
import torch.utils.data as data
import copy
import time
import tqdm
import logging
from ..tools import generate_batches
logger = logging.getLogger(__name__)
from ..train_bb_encoder.utils import simulate_brownian_bridge_v2


def load_samples(datatype, datapath):
    samples = []
    story_sent_keys = ['s1', 's2', 's3', 's4', 'end']
    with open(datapath, 'r') as fin:
        for line in fin:
            d = json.loads(line)
            events = []
            for k in story_sent_keys:
                events.append(d[k])
            assert len(events) == 5
            samples.append(events)

    return samples


def encode_event_feats(event_list, encoder, tokenizer, device):
    batches = generate_batches(event_list, shuffle=False, batch_size=128)
    event2feat = dict()
    for batch in batches:
        # sentences = [sep_token+' '+event for event in batch]
        # print(batch)
        output = tokenizer(
            [' '+event for event in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        with torch.no_grad():
            cl_feats = encoder.forward(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device))  # 1, feat_size
            feats = cl_feats.cpu()
            assert len(batch) == feats.size(0)
        for i in range(len(batch)):
            event2feat[batch[i]] = feats[i]
    return event2feat


def perturb_event(event_tokens, ratio, mask_token):
    _noisy = []
    for tok in event_tokens:
        if random.random() < ratio:
            _noisy.append(tok)
        else:
            _noisy.append(mask_token)
    _res = [_noisy[0]]
    for i in range(1, len(_noisy)):
        if _noisy[i] == mask_token and _res[-1] == mask_token:
            continue
        else:
            _res.append(_noisy[i])
    return _res


def counter_event(samples):
    event_list, event_dict = [], dict()
    for events in samples:
        for event in events:
            if event not in event_dict:
                event_dict[event] = len(event_list)
                event_list.append(event)
    return event_list, event_dict


def mask_sequence(tokens, unmask_ratio, mask_token):
    indices = list(range(len(tokens)))
    for _ in range(5):
        random.shuffle(indices)
    unmasked_indices = indices[:int(len(indices)*unmask_ratio)]
    unmasked_indices = set(unmasked_indices)
    _noisy = []
    for ind in range(len(tokens)):
        tok = tokens[ind]
        if ind in unmasked_indices:
            _noisy.append(tok)
        else:
            _noisy.append(mask_token)
    _res = [_noisy[0]]
    for i in range(1, len(_noisy)):
        if _noisy[i] == mask_token and _res[-1] == mask_token:
            continue
        else:
            _res.append(_noisy[i])
    return _res


class RocStoriesBartDataset(data.Dataset):
    """
    ROC STORIES
    """

    def __init__(self, tokenizer, sep_token, samples, event2feats,
                 source_length, target_length, perturb_ratio, ):

        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length
        self.sep_token = sep_token
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        self.event2feats = event2feats
        self.samples = samples

        self.src_input_ids = []
        self.bridge_starts = []
        self.bridge_ends = []
        self.trg_input_ids = []
        self.sample_idx = []
        self.bridge_feats = []
        for sample_idx, story in enumerate(self.samples):
            self.sample_idx.append(sample_idx)
            input_tokens = []
            input_tokens.extend(self.tokenizer.tokenize(' ' + story[0]))
            middle_tokens = self.tokenizer.tokenize(' ' + ' '.join(story[1:-1]))
            masked_middle = mask_sequence(middle_tokens, perturb_ratio, tokenizer.mask_token)
            input_tokens.extend(masked_middle)
            input_tokens.extend(self.tokenizer.tokenize(' ' + story[-1]))


            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids = [self.bos_id]+input_ids+[self.eos_id]
            self.src_input_ids.append(input_ids)
            start_feat = self.event2feats[story[0].strip()]
            end_feat = self.event2feats[story[-1].strip()]


            self.bridge_starts.append(start_feat)
            self.bridge_ends.append(end_feat)

            target_events = story[1:-1]
            target_tokens, trg_feats, trg_feat_mask = [], [], []
            for i, event in enumerate(target_events):
                event_tokens = tokenizer.tokenize(' ' + event)
                target_tokens.extend(event_tokens)
            # print(target_tokens)
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_ids = [self.bos_id]+target_ids
            self.trg_input_ids.append(target_ids)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        src_input_ids = self.src_input_ids[i]
        src_attention_mask = [1]*len(src_input_ids)
        while len(src_input_ids) < self.source_length:
            src_input_ids.append(self.pad_id)
            src_attention_mask.append(0)

        start = self.bridge_starts[i]
        end = self.bridge_ends[i]
        # [5, dim]
        bridge_feats = simulate_brownian_bridge_v2(start, end, num_samples=5)

        trg_inputs = self.trg_input_ids[i]
        decoder_input_ids = [self.eos_id] + trg_inputs
        decoder_attention_mask = [1]*len(decoder_input_ids)
        labels = trg_inputs + [self.eos_id]
        while len(decoder_input_ids) < self.target_length:
            decoder_input_ids.append(self.pad_id)
            decoder_attention_mask.append(0)
            labels.append(-100)

        src_input_ids = torch.tensor(src_input_ids)
        src_attention_mask = torch.tensor(src_attention_mask)
        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        labels = torch.tensor(labels)
        idx = self.sample_idx[i]
        idx = torch.tensor(idx)

        res = [src_input_ids, src_attention_mask,
               bridge_feats,
               decoder_input_ids, decoder_attention_mask, labels,
               idx
               ]

        return res


def load_event_feats(samples, cl_model, cl_tokenizer, device):
    event_list = []
    for d in samples:
        event_list.append(d[0])
        event_list.append(d[-1])
    # event_list, _ = counter_event(samples)
    event2feats = encode_event_feats(event_list, cl_model, cl_tokenizer, device)
    return event2feats

