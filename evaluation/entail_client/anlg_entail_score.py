# -*- coding:utf-8  -*-


import argparse
import torch
import ujson as json
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from ..eval_client.nli_metrics.src.model import *
from tools import generate_batches

class EntailScore:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = RobertaClassifier.from_pretrained(model_path)
        self.model.to(device)
        self.device = device

    def score_from_list(self, premises, hypotheses):
        if isinstance(premises, str):
            premises = [premises]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]

        results = []
        for pri, hyp in zip(premises, hypotheses):
            encoded = self.tokenizer.encode_plus(
                pri, hyp,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
            )  # with batch size = 1
            for key in encoded.keys():
                encoded[key] = encoded[key].to(self.model.device)

            logits = self.model(**encoded)[0]
            entail_prob = logits.softmax(-1)[0][2].tolist() # 2 for entail
            results.append(entail_prob)
        return results

    def _score(self, entail_init_prob, entail_count_prob):
        return entail_count_prob

    def score_from_json(self, js, raw_prob=False):
        init_premise = js.original_context
        count_premise = js.cf_context
        hypothesis = js.predicted_ending
        entail_init_prob = self.score_from_list(init_premise, hypothesis)[0]
        entail_count_prob = self.score_from_list(count_premise, hypothesis)[0]
        if raw_prob:
            return entail_init_prob, entail_count_prob
        else:
            return self._score(entail_init_prob, entail_count_prob)

    def score_from_jsonl(self, jsl):
        '''
        return: mean, by_instance
        '''
        scorel = [self.score_from_json(x) for x in jsl]
        if len(scorel) == 0: return 0, []
        score = sum(scorel) / len(scorel)
        return score, scorel

    def score_from_samples(self, sentences):
        encodings = self.tokenizer(
            sentences, padding=True,
            return_tensors='pt', truncation=True
        ).to(self.device)

        logits = self.model(**encodings)[0]
        entail_prob = logits.softmax(-1)[:, -1].cpu().tolist()  # 2 for entail
        return entail_prob

    def score_from_batch(self, samples, raw_prob):
        premise_init_generated_items = []
        premise_coun_generated_items = []
        for js in samples:
            init_premise = js.original_context
            count_premise = js.cf_context
            hypothesis = js.predicted_ending
            premise_init_generated_items.append(' '.join((init_premise, hypothesis)))
            premise_coun_generated_items.append(' '.join((count_premise, hypothesis)))
        entail_init_prob = self.score_from_samples(premise_init_generated_items)
        entail_count_prob = self.score_from_samples(premise_coun_generated_items)
        # entail_init_prob = self.score_from_list(init_premise, hypothesis)[0]
        # entail_count_prob = self.score_from_list(count_premise, hypothesis)[0]
        if raw_prob:
            return entail_init_prob, entail_count_prob
        else:
            return self._score(entail_init_prob, entail_count_prob)

    def score_from_instances_in_batches(self, instances):
        scorel = []
        batches = generate_batches(instances, shuffle=False, batch_size=4)
        for batch in batches:
            res = self.score_from_batch(batch.tolist(), False)
            scorel.extend(res)
        if len(scorel) == 0: return 0, []
        score = sum(scorel) / len(scorel)
        return score, scorel

    def score_from_file(self, filename: str, out_filename: str, raw_prob=False):
        with open(filename) as f:
            data = f.readlines()
        with open(out_filename, 'w') as fout:
            fout.write(f'entail_initial_prob\tentail_counterfactual_prob\n')
            for x in tqdm(data):
                js = json.loads(x)
                prob1, prob2 = self.score_from_json(js, raw_prob)
                js['entail_initial_prob'] = prob1
                js['entail_counterfactual_prob'] = prob2
                if raw_prob:
                    fout.write(f'{prob1}\t{prob2}\n')
        print(f'{filename} scored.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--raw', '-r', action='store_true', help='output raw entailment prob to the file')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = EntailScore(args.model_path, device)
    scorer.score_from_file(args.input, args.output, args.raw)


if __name__ == '__main__':
    main()
