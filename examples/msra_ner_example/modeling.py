import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import config as Conf

from transformers import ElectraModel
from transformers.modeling_electra import ElectraPreTrainedModel

class ElectraForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        """
        Token分类器
        :param config: 初始化的ElectraConfig
        """
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None,labels=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None):
        """
        一个batch_size的前向过程
        :param input_ids: batch_size大小的tokens id,  最大序列长度, shape: torch.Size([12, 160]), [batch_size, max_seq_len]
        :param attention_mask: 对应的attention mask, torch.Size([12, 160])
        :param labels: torch.Size([12, 160])， 因为是token级别预测，预测每个token的类别
        :param token_type_ids: None
        :param position_ids: None
        :param head_mask: None
        :param inputs_embeds: None
        :return:
        """
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        output = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            output = (loss,) + output

        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)

def ElectraForTokenClassificationAdaptorTraining(batch, model_outputs):
    return {'losses':(model_outputs[0],)}

def ElectraForTokenClassificationAdaptor(batch, model_outputs):
    return {'logits':(model_outputs[1],),
            'hidden':model_outputs[2],
            'input_mask':batch[1],
            'logits_mask':batch[1]}
