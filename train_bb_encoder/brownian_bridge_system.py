# -*-coding:utf-8-*-
from transformers import (
    GPT2Model, BartModel,
BartPretrainedModel,BartConfig
)
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from typing import Optional, Tuple, Union, List
import torch
import torch.utils.checkpoint
from torch import nn

import os
import torch
from torch import nn
import pytorch_lightning as pl
from encoder_datahelper import RocStoriesTriplet
from utils import (
    set_tokenizer, simulate_brownian_bridge_v2,
    create_dataloader, calculate_bridge_distance
)
import json
from transformers import BartTokenizer, GPT2Tokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput

torch.autograd.set_detect_anomaly(True)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False


class LatentOUEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, new_num_tokens, model_type, finetune=False):
        super(LatentOUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune
        self.new_num_tokens = new_num_tokens
        self.model_type = model_type.lower()
        self._init_model()

    def _init_model(self):

        self.model = BartModel.from_pretrained('facebook/bart-base')
        self.embedding_dim = self.model.shared.embedding_dim

        if not self.finetune:
            # turn off all the gradients
            for p in self.model.parameters():
                p.requires_grad = False
            self.model = self.model.eval()

        self.mlp = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.feature_extractor = self.create_feature_extractor()  # data_dim -> hidden_dim
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)
        self.mlp.apply(weights_init)
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        ])

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
        ])

    def get_embeddings(self, input_ids, attention_mask):
        encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

        _emb = encoder_hidden_states[0]
        bart_emb = self.compute_masked_means(_emb, attention_mask)
        return bart_emb

    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, _emb):
        z = self.mlp(_emb)  # 32, 100
        z = self.feature_extractor(z)
        return z

    def forward(self, input_ids, attention_mask):
        _emb = self.get_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        # Index into the last hidden state of the sentence (last non-EOS token)
        # gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(_emb)


class BartTextEncoder(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "train_decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, hidden_dim, latent_dim):
        super().__init__(config)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.mlp = nn.Linear(config.d_model, self.hidden_dim)
        self.feature_extractor = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        ])

        self.log_q = nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
        ])
        self.C_eta = nn.Linear(1, 1)
        # NEW AUG 19, turn off bias training.
        self.mlp.apply(weights_init)
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

        # Initialize weights and apply final processing
        self.post_init()

    def get_log_q(self, x):
        return self.log_q(x)

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, _emb):
        z = self.mlp(_emb)  # 32, 100
        z = self.feature_extractor(z)
        return z

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        _emb = encoder_outputs[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        bart_emb = self.compute_masked_means(_emb, attention_mask)
        # Index into the last hidden state of the sentence (last non-EOS token)
        # gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(bart_emb)


class BrownianBridgeLoss(object):
    """Everything is a brownian bridge...

    p(z_t | mu_0, mu_T) = \mathcal{N}(mu_0 * t/T + mu_T * (1-t/T), I t*(T-t)/T)

    normalization constant: -1/(2 * t*(T-t)/T)
    """

    def __init__(self,
                 z_0, z_t, z_T,
                 t_, t, T,
                 alpha, var,
                 log_q_y_T,
                 loss_type,
                 eps,
                 max_seq_len,
                 C_eta=None,
                 label=None):
        super().__init__()
        self.log_q_y_T = log_q_y_T
        self.z_0 = z_0
        self.z_t = z_t
        self.z_T = z_T
        self.t_ = t_
        self.t = t
        self.T = T
        self.alpha = alpha
        self.var = var
        NAME2LOSS = {
            'simclr': self.simclr_loss,
        }
        self.loss_f = NAME2LOSS[loss_type]
        self.eps = eps
        self.max_seq_len = max_seq_len
        self.sigmoid = nn.Sigmoid()
        self.label = label

        if C_eta is None:
            C_eta = 0.0
        self.C_eta = C_eta
        self.end_pin_val = 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _log_p(self, z_0, z_t, z_T, t_0, t_1, t_2):
        T = t_2 - t_0
        t = t_1 - t_0

        alpha = (t / (T + self.eps)).view(-1, 1)
        delta = z_0 * (1 - alpha) + z_T * (alpha) - z_t
        var = (t * (T - t) / (T + self.eps))
        log_p = -1 / (2 * var + self.eps) * (delta * delta).sum(-1) + self.C_eta  # (512,)
        if len(log_p.shape) > 1:  # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, z_0, z_T, z_t, t_, t, T):
        """
        Calculating log p(z_tp1, z_t) = -|| h(z_{t+dt}) - h(z_t)(1-dt)||^2_2
        """
        log_p = self._log_p(z_0=z_0, z_t=z_t, z_T=z_T,
                            t_0=t_, t_1=t, t_2=T)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_T
        logit = log_p  # - log_q
        return logit  # should be (bsz, 1)

    def reg_loss(self):
        loss = 0.0
        mse_loss_f = nn.MSELoss()
        # start reg
        start_idxs = torch.where((self.t_) == 0)[0]
        if start_idxs.nelement():
            vals = self.z_0[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        # end reg
        end_idxs = torch.where((self.T) == self.max_seq_len - 1)[0]
        if end_idxs.nelement():
            vals = torch.abs(self.z_T[end_idxs, :])
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device) * self.end_pin_val)
            loss += end_reg
        return loss

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(z_0=self.z_0, z_T=self.z_T, z_t=self.z_t,
                                t_=self.t_, t=self.t, T=self.T)
        pos_probs = torch.exp(pos_logit)  # (bsz,1)
        for idx in range(self.z_T.shape[0]):
            # Negative pair: logits over all possible contrasts
            # Nominal contrast for random triplet - contrast from in between
            neg_i_logit = self._logit(
                z_0=self.z_0, z_T=self.z_T, z_t=self.z_t[idx],
                t_=self.t_, t=self.t[idx], T=self.T)
            neg_i_probs = torch.exp(neg_i_logit)  # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.z_T.shape[0]
        # Regularization for pinning start and end of bridge
        reg_loss = self.reg_loss()
        loss += reg_loss
        return loss

    def get_loss(self):
        return self.loss_f()


class BrownianBridgeSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cl_eos_str = self.config.data_params.cl_eos_str
        self.tokenizer_name = self.config.data_params.language_encoder

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self._set_dataset()
        self._set_language_encoder()

        print('initialize from facebook/bart-base')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, shuffle=True)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)

    def _set_dataset(self):
        self.train_dataset = RocStoriesTriplet(
            split='train',
            tokenizer_name=self.tokenizer_name,
            tokenizer=self.tokenizer,
            seed=self.config.data_params.data_seed,
            datatype=self.config.data_params.name
            # cl_eos_str=self.cl_eos_str,
            # cl_eos_id=self.cl_eos_id
        )
        self.val_dataset = RocStoriesTriplet(
            split='val',
            tokenizer_name=self.tokenizer_name,
            tokenizer=self.tokenizer,
            seed=self.config.data_params.data_seed,
            datatype=self.config.data_params.name

            # cl_eos_str=self.cl_eos_str,
            # cl_eos_id=self.cl_eos_id
        )

    def _set_language_encoder(self):
        self.model = LatentOUEncoder(
            hidden_dim=self.config.model_params.hidden_size,
            latent_dim=self.config.model_params.latent_dim,
            new_num_tokens=len(self.tokenizer),
            model_type=self.tokenizer_name,
            finetune=False,
        )

    def forward(self, input_ids, attention_mask):
        feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_feats(self, obs):
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs, device=self.device)
        input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        feats_0 = self.get_feats(obs_0)
        feats_t = self.get_feats(obs_t)
        feats_T = self.get_feats(obs_T)

        log_q_y_tp1 = self.model.get_log_q(feats_t)
        loss_fn = BrownianBridgeLoss(
            z_0=feats_0,
            z_t=feats_t,
            z_T=feats_T,
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            log_q_y_T=log_q_y_tp1,
            loss_type=self.config.loss_params.name,
            eps=self.config.model_params.eps,
            max_seq_len=batch['total_t'].float(),
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, batch_idx)
        self.log('train_loss', loss,prog_bar=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch=batch, batch_idx=batch_idx)
        self.log('val_loss', loss,prog_bar=True,sync_dist=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss, prog_bar=True,sync_dist=True)
        tt_path = 'timetravel path'
        fname = os.path.join(tt_path, 'timetravel', "dev_data_original_end_splitted.json")
        bridge_distance = 0
        total = 0
        with open(fname, 'r') as fin:
            for line in fin:
                d = json.loads(line)
                premise = d['premise']
                counterfactual = d['counterfactual']
                edited_endings = d['edited_endings']
                cf_end = edited_endings[0]
                sentences = [
                    premise, counterfactual, cf_end[0], cf_end[1], cf_end[2]
                ]

                with torch.no_grad():
                    story_len = 5
                    cl_input_ids, cl_attention_mask = self.train_dataset.tokenize_caption(sentences, self.device)
                    real_bridge = self.model.forward(
                        input_ids=cl_input_ids,
                        attention_mask=cl_attention_mask)  # 1, feat_size
                    start = real_bridge[0]
                    end = real_bridge[-1]
                    s = 0
                    t = 0
                    for _ in range(5):
                        predicted_bridge = simulate_brownian_bridge_v2(start, end, num_samples=story_len)
                        dis = calculate_bridge_distance(predicted_bridge[1:story_len - 1, :],
                                                        real_bridge[1:story_len - 1, :])
                        s += dis.item()
                        t += 1
                    avg = s/t
                    bridge_distance += avg
                    total += 1
        ave_bridge_distance = bridge_distance/total
        self.log('val_bridge_loss', ave_bridge_distance, prog_bar=True,sync_dist=True)

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))


def load_cl_model(filepath, latent_dim, hidden_dim, token_size, model_type):
    model = LatentOUEncoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        new_num_tokens=token_size,
        model_type=model_type,
        finetune=False)

    state_dict = torch.load(filepath)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
            new_dict[k[6:]] = v
        elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
            continue
        elif "model." in k:
            new_dict[k[6:]] = v
        else:
            new_dict[k] = v


    model.load_state_dict(new_dict)

    return model


def get_checkpoint(latent_dim, hidden_dim, token_size, model_type, filepath=None, device=None):
    model = load_cl_model(filepath, latent_dim, hidden_dim, token_size=token_size, model_type=model_type)
    model.to(device)
    model = model.eval()
    return model
