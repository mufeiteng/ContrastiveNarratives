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
logger = logging.getLogger(__name__)


class RocStoriesTriplet(data.Dataset):
    def _set_section_names(self):
        PJ_PATH = 'yourpath'

        self.data_dir = os.path.join(PJ_PATH, 'rocstories')
        if self.split == 'train':
            self.fname = os.path.join(self.data_dir, "rocstories_train_whole.txt")
        elif self.split == 'test':
            self.fname = os.path.join(self.data_dir, "rocstories_test.txt")
        else:
            self.fname = os.path.join(self.data_dir, "rocstories_dev.txt")

    def __init__(
            self,
            split,
            tokenizer_name,
            tokenizer,
            seed,
            datatype,
            # cl_eos_str, cl_eos_id
    ):
        super().__init__()
        self.datatype = datatype
        self.split = split
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        # self.cl_eos_str = cl_eos_str
        # self.cl_eos_id = cl_eos_id
        self.max_length = 128
        self._set_section_names()
        self._process_data()

    def tokenize_caption(self, sentences, device):

        output = self.tokenizer(
            sentences,
            padding=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)

        attention_mask = output['attention_mask'].squeeze(0)


        return input_ids.to(device), attention_mask.to(device)

    def _process_data(self):
        self.processed_data = []
        doc_id = 0
        with open(self.fname, 'r') as fin:
            for line in fin:
                d = json.loads(line)

                story = []
                for k in ['s1', 's2', 's3', 's4', 'end']:
                    s = d[k].strip()
                    story.append(s)
                for sentence_counter, sentence in enumerate(story):
                    sentence_info = {
                        "sentence": sentence,
                        "sentence_id": sentence_counter,
                        "doc_id": doc_id,
                        "total_doc_sentences": len(story)
                    }
                    self.processed_data.append(sentence_info)
                doc_id += 1

        print("Examples: {}".format(self.processed_data[10]))

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        T = sentence_num
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

