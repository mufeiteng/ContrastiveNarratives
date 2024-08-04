# -*-coding:utf-8-*-
from evaluation.bleu_custom import compute_bleu, sentence_bleu_scorer
from evaluation.rouge_custom import corpus_rouge_moses
from evaluation.distinct import distinct_n_corpus_level
from evaluation.meteor_custom import corpus_meteor_moses
from nltk import word_tokenize


def eval_from_json_samples(samples):
    predictions, references = [], []
    for d in samples:
        source = d['source']
        target = d['target']
        generated = d['generated']
        predictions.append(word_tokenize(generated))
        references.append([word_tokenize(target)])

    meteor = corpus_meteor_moses(list_of_refs=references, list_of_hypos=predictions)
    bleu2 = compute_bleu(reference_corpus=references, translation_corpus=predictions, max_order=2)[0]
    bleu4 = compute_bleu(reference_corpus=references, translation_corpus=predictions, max_order=4)[0]
    dis1 = distinct_n_corpus_level(predictions, 1)
    dis2 = distinct_n_corpus_level(predictions, 2)
    rouges = corpus_rouge_moses(list_of_hypos=predictions, list_of_refs=references)
    rougel = rouges[-1]
    metrics = {
        'bleu2': bleu2, 'bleu4': bleu4, 'dis1': dis1, 'dis2': dis2,
        'rougel': rougel,
    }
    res = dict()
    for k in metrics:
        res[k] = '{:.4f}'.format(metrics[k])
    return res
