# -*-coding:utf-8-*-
from ..tools import load_json_samples, project_data_path
import os
import rouge
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

import torch
from bert_score import score as bert_score
import edlib


def _clean_text(txt):
    return txt.lower()


def eval_bleu(references, hypotheses):
    corpus_bleu_scores2 = corpus_bleu(
        references, hypotheses, smoothing_function=SmoothingFunction().method4, weights=(0.5, 0.5, 0, 0)
    )
    corpus_bleu_scores4 = corpus_bleu(
        references, hypotheses, smoothing_function=SmoothingFunction().method4, weights=(0.25, 0.25, 0.25, 0.25)
    )
    # bleu4 = compute_bleu(reference_corpus=references, translation_corpus=hypotheses, max_order=4)[0]
    res = {
        'bleu2': corpus_bleu_scores2,
        'bleu4': corpus_bleu_scores4,
        # 'bleu4': bleu4
        # 'mean_sentence_bleu': np.mean(sentence_bleu_scores),
        # 'sentence_bleu_by_instance': sentence_bleu_scores
    }
    return res


def eval_rouge(references, hypotheses):
    # instances = process_data(pred_endings_file, gold_file)

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    refers_text_list = []
    hypos_text_list = []
    for i in range(len(references)):
        _r = [_clean_text(' '.join(g)) for g in references[i]]
        _h = _clean_text(' '.join(hypotheses[i]))
        refers_text_list.append(_r)
        hypos_text_list.append(_h)

    scores = evaluator.get_scores(hypos_text_list, refers_text_list)
    # print(scores)
    # rouges = corpus_rouge_moses(list_of_hypos=hypotheses, list_of_refs=references)
    return {'rouge_l': scores['rouge-l']['f'],
            # 'rouges_v2': rouges
            }


def eval_bert_score(references, hypotheses, bert_model="bert-base-uncased"):
    # instances = process_data(pred_endings_file, gold_file)
    refers, hypos = [], []
    for i in range(len(references)):
        clean_hypothesis = _clean_text(hypotheses[i])
        if len(clean_hypothesis) == 0:
            continue
        for gold_end in references[i]:
            clean_reference = _clean_text(gold_end)
            refers.append(clean_reference)
            hypos.append(clean_hypothesis)

    P, R, F1 = bert_score(hypos, refers, model_type=bert_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    return {
        # "bert_score_P": P.mean().item(),
        # "bert_score_R": R.mean().item(),
        "bert_F1": F1.mean().item(),
        # "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        # "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        # "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def eval_nli(samples, ent_model_path, batch_size=16, verbose=False):
    # model_path = f'{os.environ["PJ_HOME"]}/models/nli_metrics/roberta-large'
    from eval_client.nli_metrics.entail_score import EntailScore
    nli_scorer = EntailScore(ent_model_path)

    # pred_endings = read_lines(filename=pred_endings_file)
    # gold_records = read_jsonl_lines(gold_file)

    for d in samples:
        d['generated_endings'] = d['predicted']
    # for js, pred in zip(gold_records, pred_endings):
    #     js['generated_endings'] = pred

    score, score_by_instance = nli_scorer.score_from_jsonl(samples, bs=batch_size, verbose=verbose)
    res = {'entail_score': score,
           # 'entail_score_by_instance': score_by_instance
           }
    return res

def eval_edit_distance(references, hypotheses):
    scores = []
    for i in range(len(references)):
        for gold_end in references[i]:
            res = edlib.align(gold_end, hypotheses[i], )
            dist = res['editDistance']
            scores.append(dist/100)
    return {'minedit': sum(scores) / len(scores)}


def eval_cfstory_from_file(path, gold_first):
    samples = load_json_samples(path)
    return eval_cfstory_from_samples(samples, gold_first)


def get_hmean(bleu4, entscore):
    return 2*bleu4*entscore/(bleu4+entscore)


def eval_cfstory_from_samples(samples, gold_first, eval_entscore=False, bs=16):

    references, hypothesis = [], []
    refer_texts, hypo_texts = [], []
    for i in range(len(samples)):
        d = samples[i]
        edited_endings = d['edited_endings']
        gene_end = d['predicted']
        assert len(edited_endings) == 3
        if gold_first:
            gold_ends_first_sents = []
            for end in edited_endings:
                assert len(end) == 3
                gold_ends_first_sents.append(word_tokenize(end[0]))
            references.append(gold_ends_first_sents)
            hypothesis.append(word_tokenize(gene_end))
            refer_texts.append([end[0] for end in edited_endings])
            hypo_texts.append(gene_end)
        else:
            gold_ends_sents = []
            for end in edited_endings:
                assert len(end) == 3
                gold_ends_sents.append(word_tokenize(' '.join(end)))
            references.append(gold_ends_sents)
            hypothesis.append(word_tokenize(gene_end))
            refer_texts.append([' '.join(end) for end in edited_endings])
            hypo_texts.append(gene_end)
    metrics = dict()
    bleu = eval_bleu(references=references, hypotheses=hypothesis)
    # print(res)
    metrics.update(bleu)
    rouge = eval_rouge(references=references, hypotheses=hypothesis)
    # print(res)
    metrics.update(rouge)
    berts = eval_bert_score(references=refer_texts, hypotheses=hypo_texts)
    # print(res)
    metrics.update(berts)
    editscore = eval_edit_distance(references=refer_texts, hypotheses=hypo_texts)
    # print(score)
    metrics.update(editscore)
    if eval_entscore:
        ent_path = os.path.join(project_data_path, 'timetravel/cfstory_nli_metrics/roberta-large')
        ent_score = eval_nli(samples, ent_model_path=ent_path, verbose=True, batch_size=bs)
        metrics.update(ent_score)
        metrics.update({'h-mean': get_hmean(bleu['bleu4'], ent_score['entail_score'])})
    for k in metrics:
        v = metrics[k]
        metrics[k] = '{:.4f}'.format(float(v) * 100)
    return metrics

