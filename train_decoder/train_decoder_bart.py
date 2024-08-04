
import os
import torch
import math
import json
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed, DistributedType
import logging
from transformers import BartTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartConfig, get_scheduler, AdamW, GPT2Config
from model_brownian_generator_bart import BartForConditionalGeneration
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from tqdm import tqdm
import time
import argparse
import numpy as np
import pickle
import shutil


from decoder_datahelper import (
    RocStoriesBartDataset, CoherenceDataset, FluencyDataset,
    load_samples, load_event_feats
)

from ..evaluation.eval_seg import eval_from_json_samples
from ..tools import JsonDumpHelper, split_three_sentence
logger = get_logger(__name__)


def set_tokenizer(tokenizer_name, cl_eos_str):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    tokenizer.add_tokens([cl_eos_str])
    cl_eos_id = tokenizer(cl_eos_str, add_special_tokens=False)['input_ids'][0]
    return tokenizer, cl_eos_id


gradient_accumulation_steps = 1
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=gradient_accumulation_steps)
# Initialize accelerator
if accelerator.distributed_type == DistributedType.TPU and gradient_accumulation_steps > 1:
    raise NotImplementedError(
        "Gradient accumulation on TPUs is currently not supported. Pass `gradient_accumulation_steps=1`"
    )

global_step, tr_loss, logging_loss = 0, 0, 0

MODEL_CLASSES = {
    'bart': (BartConfig, BartForConditionalGeneration),
}


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



def simulate_brownian_bridge_v2(B_0, B_T, num_samples):
    def get_std(t, T):
        return math.sqrt(t * (T - t) / T)

    bridge = [B_0]
    T = num_samples
    for t in range(1, T - 1):
        std = get_std(t, T)
        mean = (1 - t / T) * B_0 + (t / T) * B_T
        noise = torch.normal(mean=0, std=1, size=B_0.size(), device=B_0.device)
        hidden = std * noise + mean
        bridge.append(hidden)
    bridge.append(B_T)
    bridge = torch.stack(bridge, dim=0)
    return bridge

@torch.no_grad()
def validate_step(args, tokenizer, model, eval_dataset, device):
    model.eval()
    for i in range(2):
        input_ids = eval_dataset.src_input_ids[i]
        bridge_start = eval_dataset.bridge_starts[i]
        bridge_end = eval_dataset.bridge_ends[i]
        attention_mask = [1] * len(input_ids)
        bridge_feats = simulate_brownian_bridge_v2(bridge_start, bridge_end, 5)
        bridge_feats = bridge_feats.unsqueeze(0)

        output_sequences = model.generate(
            input_ids=torch.tensor([input_ids]).to(device),
            attention_mask=torch.tensor([attention_mask]).to(device),
            bridge_feats=bridge_feats.to(device),
            num_beams=1,
            max_length=args.target_len,
        )
        trg_input_ids = eval_dataset.trg_input_ids[i]
        logger.info('source--->'+tokenizer.decode(input_ids, skip_special_tokens=False))
        logger.info('target--->'+tokenizer.decode(trg_input_ids, skip_special_tokens=False))
        text = tokenizer.batch_decode(output_sequences.cpu().numpy().tolist(), skip_special_tokens=False)
        logger.info(text)
        logger.info('\n')
        # exit(1)
        # break



@torch.no_grad()
def generation_epoch(args, tokenizer, model, eval_dataset, epoch, device):
    raw_samples = eval_dataset.samples
    # eval_dataset = accelerator.prepare(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)
    # eval_dataloader = accelerator.prepare(eval_dataloader)
    model.eval()

    generated_instances = []
    _result = []
    for batch in eval_dataloader:
        # batch = tuple(b.to(device) for b in batch)
        src_input_ids, src_attention_mask, \
        bridge_feats, \
        decoder_input_ids, decoder_attention_mask, labels, idx = batch
        output_ids = model.generate(
            input_ids=src_input_ids.to(device),
            attention_mask=src_attention_mask.to(device),
            bridge_feats=bridge_feats.to(device),
            num_beams=1,
            max_length=args.target_len,
        )

        output_ids = output_ids.cpu().numpy().tolist()
        decoder_input_ids = decoder_input_ids.cpu().numpy().tolist()
        idx = idx.cpu().numpy().tolist()
        for j in range(len(idx)):
            d = raw_samples[idx[j]]
            generated = tokenizer.decode(output_ids[j], skip_special_tokens=False)
            gold = tokenizer.decode(decoder_input_ids[j], skip_special_tokens=False)

            item = {
                'raw': d,
                'gold': gold,
                'generated': generated
            }
            generated_instances.append(item)
            generated = tokenizer.decode(output_ids[j], skip_special_tokens=True)
            gold = tokenizer.decode(decoder_input_ids[j], skip_special_tokens=True)
            # print(d)
            evald = {
                'source': ' '.join(d),
                'target': gold,
                'generated': generated
            }
            _result.append(evald)
        # exit(1)

    out_file = os.path.join(args.output_dir, 'generated_dev_epoch_{}.json'.format(epoch))
    with open(out_file, 'w') as fout:
        for d in generated_instances:
            fout.write(json.dumps(d))
            fout.write('\n')

    metrics = eval_from_json_samples(_result)
    # metrics['coh'] = avg_coh
    return metrics, generated_instances


def train(args, model, tokenizer, train_samples, train_event2feats,
          val_dataset, ):
    num_update_steps_per_epoch = math.ceil(len(train_samples) / args.per_gpu_train_batch_size)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    logger.info(accelerator.state)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, no_deprecation_warning=True)

    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)


    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_samples))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size, )
    logger.info("  Total optimization steps = %d", num_training_steps)

    def _train_epoch():
        train_dataset = RocStoriesBartDataset(
            tokenizer, args.sep_token, train_samples, train_event2feats,
            args.source_len, args.target_len, args.perturb_ratio,
        )
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.per_gpu_train_batch_size, num_workers=args.num_workers)
        train_dataloader = accelerator.prepare(train_dataloader)

        global global_step, tr_loss, logging_loss
        num_devices = accelerator.num_processes
        batch_num_per_device = int(len(train_dataset) / (args.per_gpu_train_batch_size*num_devices))
        progress_bar = None
        if args.verbose:
            progress_bar = tqdm(range(batch_num_per_device + 1), disable=not accelerator.is_local_main_process)

        for batch in train_dataloader:
            model.train()
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                input_ids, attention_mask, \
                bridge_feats,\
                 decoder_input_ids, decoder_attention_mask, labels, idx = batch

                output = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    bridge_feats=bridge_feats,
                )

                loss = output[0]
                # print(loss)
                accelerator.backward(loss)
                # checks whether the gradients are currently being synced across all processes.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                if args.verbose:
                    progress_bar.update(1)
                tr_loss += loss.detach().item()
                global_step += 1

            if global_step % args.logging_steps == 0:
                # if accelerator.is_main_process:
                logger.info(
                    "Step: {} | Loss: {:.10f}".format(global_step, (tr_loss - logging_loss) / args.logging_steps),
                )
                logging_loss = tr_loss

            if global_step % args.validate_steps == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                validate_step(args, tokenizer, unwrapped_model, val_dataset, accelerator.device)

    best_acc = 0
    for epoch in range(int(args.num_train_epochs)):
        start_time = time.time()
        _train_epoch()
        accelerator.wait_for_everyone()

        logger.info('=============Evaluation=============')
        unwrapped_model = accelerator.unwrap_model(model)
        dev_result, dev_generated = generation_epoch(
            args, tokenizer, unwrapped_model, val_dataset, epoch, accelerator.device
        )
        logger.info(dev_result)

        if accelerator.is_main_process:
            dev_res = dev_result['bleu2']
            if best_acc < dev_res:
                output_dir = args.saved_path
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_dir = args.saved_path
                unwrapped_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info('model saved to {}'.format(output_dir))

            best_acc = max(dev_res, best_acc)
        end_time = time.time()
        logger.info('epoch {} use time {:.4f} mins'.format(epoch, (end_time - start_time) / 60))
        logger.info('*' * 20)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=None, type=str, required=True, )
    parser.add_argument("--val_path", default=None, type=str, required=True, )
    parser.add_argument("--test_path", default=None, type=str, required=True, )
    parser.add_argument("--datatype", default='rocstories', type=str)
    parser.add_argument("--feat_path", default='rocstories', type=str)


    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--source_len", default=128, type=int)
    parser.add_argument("--target_len", default=128, type=int)

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, )

    parser.add_argument("--model_type", default=None, type=str, required=True, )
    parser.add_argument("--sep_token", default=' [SEP]', type=str)

    parser.add_argument('--use_contrastive_embeddings', action='store_true',
                        help="Whether to train")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to train")
    parser.add_argument("--latent_dim", default=16, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--encoder_filepath", default='', type=str, required=True)
    parser.add_argument("--perturb_ratio", default=0.5, type=float)
    parser.add_argument('--verbose', action='store_true',
                        help="Whether to train")

    parser.add_argument("--fraction", default=0.1, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--l2_reg", default=0.1, type=float)

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument('--logging_steps', type=int, default=400,
                        help="Log every X updates steps.")
    parser.add_argument('--validate_steps', type=int, default=3000)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    return args



def main():
    args = get_config()
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # args.density_cached_path = os.path.join(args.output_dir, 'density_cached')
    # if not os.path.exists(args.density_cached_path):
    #     os.makedirs(args.density_cached_path)
    # transformers.utils.logging.set_verbosity_info()
    set_logger(logfile=os.path.join(args.output_dir, 'log.txt') if args.do_train else None)
    args.saved_path = os.path.join(args.output_dir, '{}-checkpoint'.format(args.model_type))


    event2feats = pickle.load(open(args.feat_path, 'rb'))
    global_event2feats = dict()
    for event in event2feats:
        t = torch.tensor(event2feats[event])
        global_event2feats[event] = t

    val_samples = load_samples(args.datatype, args.test_path)

    tokenizer, cl_sep_id = set_tokenizer(args.model_type, args.sep_token)
    logger.info("Added special tokens, {}".format(args.sep_token))
    config_class, model_class = MODEL_CLASSES[args.model_type]

    val_dataset = RocStoriesBartDataset(
        tokenizer, args.sep_token, val_samples, global_event2feats,
        args.source_len, args.target_len, args.perturb_ratio,
    )
    if args.do_train:

        model = model_class.from_pretrained(
            args.model_name_or_path,
            use_contrastive_embeddings=True,
            cl_latent_dim=args.latent_dim,
            cl_sep_id=cl_sep_id,
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model to {}".format(len(tokenizer)))
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")


        train_samples = load_samples(args.datatype, args.train_path)


        train(args, model, tokenizer, train_samples, global_event2feats,
              val_dataset=val_dataset,

              )

if __name__ == "__main__":
    main()
