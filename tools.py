import os
import codecs
import numpy as np
import json
import random
import torch
import numpy
import multiprocessing
import logging
import shutil


def generate_batches(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def load_json_samples(path, add_sample_idx=False):
    _samples = []
    with open(path, 'r') as fin:
        cnt = 0
        for line in fin:
            d = json.loads(line)
            if add_sample_idx:
                d['_sample_idx_'] = cnt
            cnt += 1
            _samples.append(d)
    return _samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def save_data(samples, outpath):
    with codecs.open(outpath, 'w', 'utf-8') as fout:
        for d in samples:
            fout.write(json.dumps(d))
            fout.write('\n')


def set_log(logger, log_file=None):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                            '%m/%d/%Y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)


class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def multiprocessing_tokenize_data(implementor, data, model_type, tokenizer, train_data_name, eval_data_name):
    workers = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool()
    filesize = len(data)
    _results = []
    for i in range(workers):
        chunk_start, chunk_end = (filesize * i) // workers, (filesize * (i + 1)) // workers
        _results.append(pool.apply_async(
            implementor, (data[chunk_start:chunk_end], model_type, tokenizer, train_data_name, eval_data_name,)
        ))
    pool.close()
    pool.join()
    _total = []
    for _result in _results:
        samples = _result.get()
        _total.extend(samples)
    assert len(_total) == filesize
    return _total


def split_three_sentence(text):
    from nltk import tokenize

    SMALL_CONST = 1e-15
    BIG_CONST = 1e10

    def _extract_a_sentence(text):
        """
        Extracts the first sentence in `text`.
        Returns the sentence and the remaining text.
        """
        # (1)
        sent_terminators = ['. ', '! ', '? ']
        min_tm_index = BIG_CONST
        for tm in sent_terminators:
            tm_index = text.find(tm)
            if tm_index == -1:
                tm_index = BIG_CONST
            min_tm_index = min(min_tm_index, tm_index)

        if min_tm_index < BIG_CONST:
            return text[:min_tm_index + 1], text[min_tm_index + 2:]

        # (2)
        sent_terminators = ['." ', '!" ', '?" ']
        for tm in sent_terminators:
            tm_index = text.find(tm)
            if tm_index == -1:
                tm_index = BIG_CONST
            min_tm_index = min(min_tm_index, tm_index)

        if min_tm_index < BIG_CONST:
            return text[:min_tm_index + 2], text[min_tm_index + 3:]

        return text, ""

    def extract_three_sentences(text):
        """
        `text` is assumed to consist of three sentences. This function
        extracts and returns the three sentences.
        """
        s1, s23 = _extract_a_sentence(text)
        s2, s3 = _extract_a_sentence(s23)
        return s1, s2, s3

    sentences = tokenize.sent_tokenize(text)
    if len(sentences) == 3:
        return sentences
    sentences = extract_three_sentences(text)
    if len(sentences) == 3:
        return sentences
    raise RuntimeError('num of sentences not equal to 3.')
