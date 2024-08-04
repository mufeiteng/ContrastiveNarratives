# -*-coding:utf-8-*-

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel



class RobertaForMultipleChoiceCustomized(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
            # item1_ids=None, item1_attention_mask=None,
            # item2_ids=None, item2_attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
            # real_input_ids=None, real_attention_mask=None,
            imp_input_ids=None,imp_attention_mask=None,
            imp_labels=None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]  # [bs*num, dim]
        pooled_output = self.dropout(pooled_output)
        # print('pooled_output', pooled_output.size())
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss1 = None
        loss2 = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(reshaped_logits, labels)
            # print(loss1)
            if imp_input_ids is not None:
                dim = pooled_output.size(-1)

                bs = input_ids.size(0)
                imp_num_choices = imp_input_ids.shape[1]
                flat_imp_input_ids = imp_input_ids.view(-1, imp_input_ids.size(-1)) if imp_input_ids is not None else None

                flat_imp_attention_mask = imp_attention_mask.view(-1, imp_attention_mask.size(
                    -1)) if imp_attention_mask is not None else None

                imp_outputs = self.roberta(
                    flat_imp_input_ids,
                    attention_mask=flat_imp_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                imp_pooled_output = imp_outputs[1]
                imp_pooled_output = self.dropout(imp_pooled_output)


                imp_logits = self.classifier(imp_pooled_output)
                reshaped_imp_logits = imp_logits.view(-1, imp_num_choices)
                # print(reshaped_imp_logits.size())
                # print(imp_labels.size())
                imp_labels = imp_labels.to(reshaped_imp_logits.device)
                loss_fct = CrossEntropyLoss()
                # exit(1)
                loss2 = loss_fct(reshaped_imp_logits, imp_labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]

            if loss2 is not None:
                output = (loss2,) + output
            if loss1 is not None:
                output = (loss1,) + output
            return output
            # return ((loss, loss2,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss1,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
