from evaluation.bleu.bleu import Bleu
from evaluation.meteor.meteor_nltk import Meteor
from evaluation.rouge.rouge import Rouge
from evaluation.cider.cider import Cider
from evaluation.bert_score.bert_score import BertScore
from evaluation.distinct import distinct_n_corpus_level
from collections import defaultdict
from argparse import ArgumentParser
import sys
import json
from nltk.tokenize import word_tokenize


class QGEvalCap:
    def __init__(self, gts, res, metrics):
        self.gts = gts
        self.res = res
        self.metrics = [m.lower() for m in metrics]

    def evaluate(self):
        scorers = []
        bleu_n = []
        for idx in [1, 2, 3, 4]:
            if 'bleu{}'.format(idx):
                bleu_n.append('Bleu_{}'.format(idx))
        if bleu_n:
            scorers.append((Bleu(4), bleu_n))

        if 'meteor' in self.metrics:
            scorers.append((Meteor(),"METEOR"))
        if 'rouge' in self.metrics:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'cider' in self.metrics:
            scorers.append((Cider(), "CIDEr"))
        if 'bertscore' in self.metrics:
            scorers.append((BertScore(), "Bert Score"))
        # =================================================
        # Compute scores
        # =================================================
        output = []
        scores_dict = {}
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    output.append(sc)
                    scores_dict[m] = str(sc)
            else:
                output.append(score)
                scores_dict[method] = score
        return scores_dict


def eval_result(sources, references, predictions, metrics, lower=False, tokenize=False):
    """
        Given a filename, calculate the metric scores for that prediction file
        isDin: boolean value to check whether input file is DirectIn.txt
    """
    pairs = []
    for tup in sources:
        pair = {}
        pair['tokenized_sentence'] = tuple(tup)
        pairs.append(pair)
    cnt = 0
    for line in references:
        pairs[cnt]['tokenized_question'] = line
        cnt += 1
    output = predictions
    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = pair['prediction']
        gts[key].extend(pair['tokenized_question'])
    QGEval = QGEvalCap(gts, res, metrics=metrics)
    return QGEval.evaluate()


def do_eval_from_json_samples(samples, metrics):
    srcs = []
    references = []
    predictions = []
    for d in samples:
        obs1 = d['obs1']
        obs2 = d['obs2']
        hypo = d['hypo']
        pred = d['pred']
        srcs.append((obs1, obs2))
        references.append([hypo])
        predictions.append(pred)
    predictions = [[x] for x in predictions]
    score_dict = eval_result(
        sources=srcs, references=references,
        predictions=predictions, metrics=metrics
    )
    if 'dist2' in metrics:
        dis2 = distinct_n_corpus_level([word_tokenize(p[0]) for p in predictions], 2)
        score_dict['dist-2'] = dis2
    if 'dist3' in metrics:
        dis3 = distinct_n_corpus_level([word_tokenize(p[0]) for p in predictions], 3)
        score_dict['dist-3'] = dis3
    for k in score_dict:
        v = score_dict[k]
        score_dict[k] = '{:.4f}'.format(float(v)*100)
    return score_dict


def eval_nli(samples, ent_model_path, batch_size=16, verbose=False):
    from evaluation.eval_client.nli_metrics.entail_score import EntailScore
    nli_scorer = EntailScore(ent_model_path)
    score, score_by_instance = nli_scorer.score_from_jsonl(samples, bs=batch_size, verbose=verbose)
    return {
        'entail_score': score,
        # 'entail_score_by_instance': score_by_instance
    }


def do_eval_from_json_samples_merged_hypos(samples, metrics):
    srcs = []
    references = []
    predictions = []
    instances = []
    for d in samples:
        obs1 = d['obs1']
        obs2 = d['obs2']
        hypos = d['hypos']
        pred = d['pred']
        for hypo in hypos:
            srcs.append((obs1, obs2))
            references.append([hypo])
            predictions.append(pred)
            instances.append({'obs1': obs1, 'obs2': obs2, 'hypo': pred})

    predictions = [[x] for x in predictions]
    score_dict = eval_result(
        sources=srcs, references=references,
        predictions=predictions, metrics=metrics
    )
    if 'dist2' in metrics:
        dis2 = distinct_n_corpus_level([word_tokenize(p[0]) for p in predictions], 2)
        score_dict['dist-2'] = dis2
    if 'dist3' in metrics:
        dis3 = distinct_n_corpus_level([word_tokenize(p[0]) for p in predictions], 3)
        score_dict['dist-3'] = dis3
    if 'entail' in metrics:
        import os
        from ..tools import project_data_path
        ent_path = os.path.join(project_data_path, 'timetravel/cfstory_nli_metrics/roberta-base')

        ent_score = eval_nli(instances, ent_path, batch_size=16, verbose=True)
        score_dict.update(ent_score)
    for k in score_dict:
        v = score_dict[k]
        score_dict[k] = '{:.4f}'.format(float(v)*100)
    return score_dict

