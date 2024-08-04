# -*-coding:utf-8-*-

import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eventsim_dict", default=None, type=str, required=True, )
    parser.add_argument("--trade_off", default=0.5, type=float)
    parser.add_argument("--num_sample", default=2, type=int)

    parser.add_argument("--negsample_start", default=16, type=int)
    parser.add_argument("--negsample_end", default=16, type=int)
    parser.add_argument("--swag_data_dir", default=None, type=str, required=True, )

    parser.add_argument("--hellaswag_data_dir", default=None, type=str, required=True, )

    parser.add_argument("--copa_data_dir", default=None, type=str, required=True, )
    parser.add_argument("--ecare_data_dir", default=None, type=str, required=True, )
    parser.add_argument("--cloze_data_dir", default=None, type=str, required=True, )
    parser.add_argument("--anli_data_dir", default=None, type=str, required=True, )
    parser.add_argument("--cfstory_data_dir", default=None, type=str, required=True, )
    parser.add_argument("--output_dir", default='', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="random seed for initialization")

    parser.add_argument("--max_seq_length", default=16, type=int)
    parser.add_argument("--model_type", default=None, type=str, required=True, )
    parser.add_argument("--model_size", default=None, type=str, required=True, )
    parser.add_argument("--model_checkpoint_path", default='', type=str )


    parser.add_argument('--do_train', action='store_true',
                        help="Whether to train")
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


