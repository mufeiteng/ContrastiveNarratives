# -*- coding: utf-8 -*-
import os
import torch
import math
import json
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed, DistributedType
from ctrltextgen.tools import JsonDumpHelper
from ctrltextgen.brownianbridge.utils import set_tokenizer
import logging
from transformers import BartTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from ..train_bb_encoder.brownian_bridge_system import load_cl_model

import time
import argparse
import numpy as np
import pickle
import shutil

from decoder_datahelper import (
    load_samples, load_event_feats
)
from torch.utils.data import Dataset, DataLoader

logger = get_logger(__name__)


gradient_accumulation_steps = 1
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=gradient_accumulation_steps)
# Initialize accelerator
if accelerator.distributed_type == DistributedType.TPU and gradient_accumulation_steps > 1:
    raise NotImplementedError(
        "Gradient accumulation on TPUs is currently not supported. Pass `gradient_accumulation_steps=1`"
    )


def set_logger(logfile=None):
    console = logging.StreamHandler()
    handlers = [console]
    if logfile:
        file_handler = logging.FileHandler(logfile, "w")
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s: %(name)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers
    )


class ClFeatDataset(Dataset):
    def __init__(self, events, tokenizer, max_len=32):
        self.max_len = max_len
        self.input_ids = []
        self.events = []
        self.event_idx = []
        for i, event in enumerate(events):
            event_ids = tokenizer.encode(' '+event, add_special_tokens=True)
            self.input_ids.append(event_ids)
            self.events.append(event)
            self.event_idx.append(i)

        self.pad_id = tokenizer.pad_token_id
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        evt_idx = self.event_idx[item]
        input_ids = self.input_ids[item]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        attention_mask = [1]*len(input_ids)
        while len(input_ids)< self.max_len:
            input_ids.append(self.pad_id)
            attention_mask.append(0)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        evt_idx = torch.tensor(evt_idx)
        return input_ids, attention_mask, evt_idx



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=None, type=str, required=True, )
    parser.add_argument("--val_path", default=None, type=str, required=True, )
    parser.add_argument("--test_path", default=None, type=str, required=True, )
    parser.add_argument("--datatype", default='rocstories', type=str)
    parser.add_argument("--output_path", default='rocstories', type=str)

    parser.add_argument("--latent_dim", default=16, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, )

    parser.add_argument("--model_type", default=None, type=str, required=True, )

    parser.add_argument("--encoder_filepath", default='', type=str, required=True)

    parser.add_argument('--verbose', action='store_true',
                        help="Whether to train")


    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args



def main():
    args = get_config()

    device = torch.device('cuda:{}'.format(args.gpu_id))
    set_seed(args.seed)
    set_logger(logfile=None)

    cl_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    cl_model = load_cl_model(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        token_size=len(cl_tokenizer), model_type=args.model_type,
        filepath=args.encoder_filepath
    )

    logger.info('load cl-model from {}'.format(args.encoder_filepath))
    logger.info('calculate dev feat...')

    val_samples = load_samples(args.datatype, args.val_path)
    test_samples = load_samples(args.datatype, args.test_path)
    train_samples = load_samples(args.datatype, args.train_path)#[:1000]
    all_samples = train_samples+val_samples+test_samples

    event_list = set()
    for sample in all_samples:
        start = sample[0].strip()
        end = sample[-1].strip()
        for event in sample:
            event_list.add(event.strip())
        # event_list.add(start)
        # event_list.add(end)
    event_list = list(event_list)
    logger.info('total number of events {}'.format(len(event_list)))

    raw_eval_dataset = ClFeatDataset(event_list, cl_tokenizer, max_len=64)
    all_events = raw_eval_dataset.events

    # eval_dataset = accelerator.prepare(raw_eval_dataset)

    eval_dataloader = DataLoader(
        raw_eval_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)

    # eval_dataloader, cl_model = accelerator.prepare(eval_dataloader, cl_model)
    cl_model.to(device)
    cl_model.eval()
    start_time = time.time()
    event2feats = dict()
    for batch in eval_dataloader:
        # batch = tuple(b.to(device) for b in batch)
        input_ids, attention_mask, event_idx = batch
        with torch.no_grad():
            cl_feats = cl_model.forward(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device)
            )  # 1, feat_size
        assert len(event_idx) == cl_feats.size(0)

        # event_idx, cl_feats = accelerator.gather_for_metrics((event_idx, cl_feats))
        logger.info(cl_feats.size(0))
        event_idx = event_idx.cpu().numpy().tolist()
        cl_feats = cl_feats.cpu().numpy().tolist()
        for i in range(len(event_idx)):
            idx = event_idx[i]
            event2feats[all_events[idx]] = cl_feats[i]
    assert len(event2feats) == len(event_list)

    for event in event_list:
        assert event in event2feats

    output_file = args.output_path

    # os.path.join(args.output_path, '{}_event2feats_multicore.pickle'.format(args.datatype))
    pickle.dump(event2feats, open(output_file, 'wb'))
    end_time = time.time()
    logger.info('use  time {} mins'.format((end_time-start_time)/60))


if __name__ == '__main__':
    main()