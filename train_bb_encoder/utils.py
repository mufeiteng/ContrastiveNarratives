# -*-coding:utf-8-*-
import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer, GPT2Tokenizer


def set_tokenizer(tokenizer_name, cl_eos_str):

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    tokenizer.add_tokens([cl_eos_str])
    cl_eos_id = tokenizer(cl_eos_str, add_special_tokens=False)['input_ids'][0]
    return tokenizer, cl_eos_id


def simulate_brownian_bridge(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    if isinstance(B_0, torch.Tensor):
        B_0 = B_0.cpu().detach().numpy()
    if isinstance(B_T, torch.Tensor):
        B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    x_t = np.copy(B_0)
    for step in range(num_samples - 2):  # number of sentences
        dim = B_0.shape[-1]
        noise = np.sqrt(dt) * sigma * np.random.normal(mu, sigma, dim)
        t = step / num_samples
        x_tp1 = x_t * (1 - dt / (1 - t)) + (dt / (1 - t)) * B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_t = x_tp1

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return bridge


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


def simulate_brownian_bridge_v3(B_0, B_T, num_samples, dt=0.05, mu=0.0, sigma=1.0):
    def get_std(t, T):
        return dt * math.sqrt(t * (T - t) / T)

    bridge = []
    T = num_samples
    for t in range(1, T - 1):
        std = get_std(t, T)
        mean = (1 - t / T) * B_0 + (t / T) * B_T
        noise = torch.normal(mean=mu, std=sigma, size=B_0.size(), device=B_0.device)
        hidden = std * noise + mean
        bridge.append(hidden)
    # bridge.append(B_T)
    # bridge = torch.stack(bridge, dim=0)
    return bridge


def calculate_bridge_distance(predicted, real):
    loss_fn = nn.MSELoss()
    distance = loss_fn(predicted, real)
    return distance.detach()


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        num_workers=0
        # config.experiment_params.data_loader_workers,
    )
    return loader

