import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    RobertaTokenizer, RobertaForMultipleChoice,
    BertTokenizer, BertForMultipleChoice,
    AdamW, get_scheduler
)
import math
import json
from typing import Union, List
from tqdm import tqdm
import time
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed, DistributedType
from datahelper import RocStoriesDataset
from config import get_config
import shutil
from nludatasets.anli_data_processor import get_anli_dataset
from nludatasets.copa_data_processor import get_copa_dataset
from nludatasets.cloze_data_processor import get_cloze_dataset
from nludatasets.ecare_data_processor import get_ecare_dataset
from nludatasets.hellaswag_data_processor import get_hellaswag_dataset
from nludatasets.timetravel_processor import TimeTravelEvalDataset
from nludatasets.swag_data_processor import get_swag_dataset
# from ctrltextgen.discriminator.library import eval_nlu_datasets
from ..tools import JsonDumpHelper, load_json_samples
import logging
import transformers
from model import RobertaForMultipleChoiceCustomized
# from transformers import RobertaForMultipleChoice
logger = get_logger(__name__)

MODEL_CLASSES = {
    'bert': (BertTokenizer, BertForMultipleChoice, 'bert-{}-uncased'),
    'roberta': (RobertaTokenizer, RobertaForMultipleChoiceCustomized, 'roberta-{}'),
}

gradient_accumulation_steps = 16
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=gradient_accumulation_steps)
# Initialize accelerator
if accelerator.distributed_type == DistributedType.TPU and gradient_accumulation_steps > 1:
    raise NotImplementedError(
        "Gradient accumulation on TPUs is currently not supported. Pass `gradient_accumulation_steps=1`"
    )

global_step, tr_loss, logging_loss = 0, 0, 0


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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)



@torch.no_grad()
def do_multichoice_evaluation(model, eval_dataset, args, device):
    num_devices = accelerator.num_processes
    batch_num_per_device = int(len(eval_dataset) / (args.per_gpu_eval_batch_size * num_devices))
    # print(num_devices, batch_num_per_device)
    progress_bar = None
    if args.verbose:
        progress_bar = tqdm(range(batch_num_per_device + 1), disable=not accelerator.is_local_main_process)

    eval_dataset = accelerator.prepare(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)


    eval_dataloader = accelerator.prepare(eval_dataloader)

    # test_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=test_sampler, batch_size=args.per_gpu_eval_batch_size,
    #     drop_last=False, num_workers=args.num_workers, pin_memory=True
    # )
    # model.eval()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = None


    # progress_bar = None
    # if args.verbose:
    #     progress_bar = tqdm(range(len(eval_dataloader)))
    accurate = 0
    num_elems = 0

    for batch in eval_dataloader:
        # batch = tuple(b.to(device) for b in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        output = model(
            input_ids,
            attention_mask=input_mask,
            # token_type_ids=segment_ids if args.model_type.startswith('bert') else None,
            return_dict=False
        )
        logits = output[0]
        logits, label_ids = accelerator.gather_for_metrics((logits, label_ids))
        # logger.info(logits.size(), label_ids.size())

        prediction = torch.argmax(logits, dim=1)
        # logger.info(prediction)
        # logger.info(label_ids)
        accurate_preds = torch.eq(prediction, label_ids)
        # logger.info(accurate_preds)
        # print()

        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()
        if args.verbose:
            progress_bar.update(1)
    # print(num_elems, accurate)
    eval_accuracy = accurate/num_elems
    return eval_accuracy


def eval_nlu_datasets(args, model, copa_dataset, ecare_dataset,
                      anli_dataset, cloze_dataset, cfstory_dataset,
                      hellaswag_dataset, swag_dataset,
                      device):
    copa_acc = do_multichoice_evaluation(model, copa_dataset, args, device=device)
    ecare_acc = do_multichoice_evaluation(model, ecare_dataset, args, device=device)
    anli_acc = do_multichoice_evaluation(model, anli_dataset, args, device=device)
    cloze_acc = do_multichoice_evaluation(model, cloze_dataset, args, device=device)
    hellaswag_acc = do_multichoice_evaluation(model, hellaswag_dataset, args, device=device)
    swag_acc = do_multichoice_evaluation(model, swag_dataset, args, device=device)
    #
    # tt_x_acc, tt_xprime_acc = eval_timetravel(model, cfstory_dataset, args, device=device)
    # copa_acc = 0
    # swag_acc = 0

    metrics = {
        'ecare': ecare_acc,
        'copa': copa_acc,
        'anli': anli_acc,
        'cloze': cloze_acc,
        'tt-x': 0,
        'tt-x-prime': 0,
        'hellaswag': hellaswag_acc,
        'swag': swag_acc,
    }
    return metrics




def train(args, model, tokenizer, copa_dataset, ecare_dataset,
              anli_dataset, cloze_dataset, cfstory_dataset,
          hellaswag_dataset, swag_dataset,):
    train_dataset = RocStoriesDataset(
        tokenizer, args.train_data_file, eventsim_path=args.eventsim_dict,
        num_neg_sampling=args.num_sample, max_seq_len=args.max_seq_length,
        negsample_start=args.negsample_start, negsample_end=args.negsample_end
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.per_gpu_train_batch_size)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    logger.info(accelerator.state)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_gpu_train_batch_size, num_workers=args.num_workers)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, no_deprecation_warning=True)

    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size, )
    logger.info("  Total optimization steps = %d", num_training_steps)

    @torch.no_grad()
    def validate_step():
        unwrapped_model = accelerator.unwrap_model(model)
        logger.info('=============Validate ANLI=============')
        dataloader = DataLoader(anli_dataset, batch_size=4, shuffle=False)
        for batch in dataloader:
            batch = tuple(b.to(accelerator.device) for b in batch)
            outputs = unwrapped_model(input_ids=batch[0],
                                      attention_mask=batch[1])
            logits = outputs[0].detach()
            logger.info(logits)
            logger.info(batch[-1])
            break

        eval_dataloader = DataLoader(cfstory_dataset, shuffle=False, batch_size=4)
        for batch in eval_dataloader:
            batch = tuple(b.to(accelerator.device) for b in batch)
            x_output = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                # return_dict=False
            )
            logits = x_output[0].detach()
            logger.info(logits)
            logger.info('reduce-max:')
            logger.info(batch[3])

            x_prime_output = model(
                input_ids=batch[4],
                attention_mask=batch[5],
                # return_dict=False
            )
            logits = x_prime_output[0].detach()
            logger.info(logits)
            logger.info('reduce-min:')
            logger.info(batch[7])

            break


    def _train_epoch():
        global global_step, tr_loss, logging_loss
        num_devices = accelerator.num_processes
        batch_num_per_device = int(len(train_dataset) / (args.per_gpu_train_batch_size*num_devices))
        progress_bar = None
        if args.verbose:
            progress_bar = tqdm(range(batch_num_per_device + 1), disable=not accelerator.is_local_main_process)

        for batch in train_dataloader:

            exp_input_ids, exp_attention_mask, exp_labels, \
                imp_input_ids, imp_att_mask, imp_label = batch
            model.train()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(
                    input_ids=exp_input_ids, attention_mask=exp_attention_mask, labels=exp_labels,

                    imp_input_ids=imp_input_ids, imp_attention_mask=imp_att_mask,
                    imp_labels=imp_label,
                    return_dict=False
                )

                loss1 = outputs[0]
                loss2 = outputs[1]

                loss = (1-args.trade_off)*loss1 + args.trade_off * loss2
                # loss, task_loss, cse_loss = outputs[:3]
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
                    "Step: {} | Loss: {:.10f}, expLoss: {:.10f}, impLoss: {:.10f}".format(
                        global_step, (tr_loss - logging_loss) / args.logging_steps,
                        # task_loss, cse_loss,
                        loss1, loss2
                    ),
                )
                logging_loss = tr_loss



    best_acc = 0
    for epoch in range(int(args.num_train_epochs)):
        start_time = time.time()
        _train_epoch()
        accelerator.wait_for_everyone()


        logger.info('=============Evaluation=============')

        accelerator.wait_for_everyone()

        metrics = eval_nlu_datasets(
            args, model, copa_dataset, ecare_dataset,
            anli_dataset, cloze_dataset, cfstory_dataset,
            hellaswag_dataset, swag_dataset,
            accelerator.device)
        logger.info(
            'COPA: {:.4f}, ECare: {:.4f}, Anli: {:.4f}, Cloze: {:.4f}, Swag: {:.4f}, HellaSwag: {:.4f}'.format(
                metrics['copa'], metrics['ecare'], metrics['anli'], metrics['cloze'],
                metrics['swag'], metrics['hellaswag']
            ))
        logger.info('TT-x: {:.4f}, tt-x-prime: {:.4f}, '.format(
            metrics['tt-x'], metrics['tt-x-prime'],
        ))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_dir = os.path.join(args.saved_path+'-epoch{}'.format(epoch))
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            unwrapped_model = accelerator.unwrap_model(model)

            unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info('model saved to {}'.format(output_dir))


        end_time = time.time()
        logger.info('epoch {} use time {:.4f} mins'.format(epoch, (end_time - start_time) / 60))
        logger.info('*' * 20)




def main():
    args = get_config()
    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Create output directory if needed
    args.output_dir = os.path.join(
        args.output_dir, '{}-{}'.format(args.model_type, args.model_size))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    # transformers.utils.logging.set_verbosity_info()
    set_logger(logfile=os.path.join(args.output_dir, 'log.txt') if args.do_train else None)

    tokenizer_class, model_class, model_name_or_path = MODEL_CLASSES[args.model_type]
    model_name_or_path = model_name_or_path.format(args.model_size)
    args.saved_path = os.path.join(args.output_dir, 'checkpoint')
    if args.do_train:
        model_path = model_name_or_path
    else:
        model_path = args.saved_path

    tokenizer = tokenizer_class.from_pretrained(model_path)

    cfstory_dataset = TimeTravelEvalDataset(
        tokenizer, os.path.join(args.cfstory_data_dir, 'test_data.json'))
    copa_dataset = get_copa_dataset(args.copa_data_dir, 'test', tokenizer, args.max_seq_length, )
    anli_dataset = get_anli_dataset(args.anli_data_dir, 'test', tokenizer, args.max_seq_length, )
    cloze_dataset = get_cloze_dataset(args.cloze_data_dir, 'test', tokenizer, args.max_seq_length)
    ecare_dataset = get_ecare_dataset(args.ecare_data_dir, 'dev', tokenizer, args.max_seq_length, )
    hellaswag_dataset = get_hellaswag_dataset(
        args.hellaswag_data_dir, 'val', tokenizer, args.max_seq_length,
    )
    swag_dataset = get_swag_dataset(
        args.swag_data_dir, 'val', tokenizer, args.max_seq_length,
    )


    model = model_class.from_pretrained(model_path)
    logger.info('Load model from {}'.format(model_path))

    if args.do_train:
        train(args, model, tokenizer, copa_dataset, ecare_dataset,
              anli_dataset, cloze_dataset, cfstory_dataset,
              hellaswag_dataset,swag_dataset,


              )


if __name__ == "__main__":
    main()
