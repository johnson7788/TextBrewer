import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import torch.nn.functional as F
import config as Conf
from pytorch_pretrained_bert.my_modeling import BertModel, BertLayerNorm
from transformers import ElectraPreTrainedModel, ElectraModel, AlbertPreTrainedModel, AlbertModel
from transformers.modeling_outputs import SequenceClassifierOutput
logger = logging.getLogger(__name__)


def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights


class BertForGLUE(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForGLUE, self).__init__()
        self.num_labels = num_labels
        output_sum = None if Conf.args.output_sum < 0 else Conf.args.output_sum
        self.bert = BertModel(config,Conf.args.output_score, output_sum)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, logits_T=None,
                attention_probs_sum_layer=None, attention_probs_sum_T=None, hidden_match_layer=None, hidden_match_T=None):
        if hidden_match_layer is not None:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True,  output_attention_layer=attention_probs_sum_layer)
            hidden_states = [sequence_output[i] for i in hidden_match_layer]
            sequence_output = sequence_output[-1]
        else:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False,  output_attention_layer=attention_probs_sum_layer)
            hidden_states = []


        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels

        if logits_T is not None or labels is not None or attention_probs_sum_T is not None:
            total_loss = 0
            hidden_losses = None
            att_losses = None
            if logits_T is not None and self.num_labels!=1:
                temp=Conf.args.temperature
                logits_T /= temp
                logits /= temp
                prob_T = F.softmax(logits_T,dim=-1)
                ce_loss = -(prob_T * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                ce_loss = ce_loss.mean() #* temp_2
                total_loss += ce_loss
            if attention_probs_sum_T:
                if Conf.args.mask_inter==1:
                    mask = attention_mask.to(attention_probs_sum[0])
                    valid_count = torch.pow(mask.sum(dim=1),2).sum()
                    att_losses = [(F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i], reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count for i in range(len(attention_probs_sum_T))]
                else:
                    att_losses = [F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i]) for i in range(len(attention_probs_sum_T))]
                att_loss = sum(att_losses) * Conf.args.att_loss_weight
                total_loss += att_loss
                #mle_loss = (F.mse_loss(start_logits,start_logits_T) + F.mse_loss(end_logits,end_logits_T))/2
                #total_loss += mle_loss
            if hidden_match_T:
                if Conf.args.mask_inter==1:
                    mask = attention_mask.to(hidden_states[0])
                    valid_count = mask.sum() * hidden_states[0].size(-1)
                    hidden_losses = [(F.mse_loss(hidden_states[i],hidden_match_T[i], reduction='none')*mask.unsqueeze(-1)).sum() / valid_count for i in range(len(hidden_match_layer))]
                else:
                    hidden_losses = [F.mse_loss(hidden_states[i],hidden_match_T[i]) for i in range(len(hidden_match_layer))]
                hidden_loss = sum(hidden_losses) * Conf.args.hidden_loss_weight
                total_loss += hidden_loss

            if labels is not None:
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.view(-1), labels.view(-1))
                else:
                    loss = F.cross_entropy(logits,labels)
                total_loss += loss
            return total_loss, att_losses, hidden_losses
        else:
            if attention_probs_sum_layer is not None or hidden_match_layer is not None:
                return logits, attention_probs_sum, hidden_states
            else:
                return logits, None

class BertSPCSimple(nn.Module):
    def __init__(self, config, num_labels, args):
        """
        简单的分类任务
        :param config: 模型的配置 从config.json加载
        :param num_labels: int   eg ：3
        :param args:
        """
        super(BertSPCSimple, self).__init__()
        self.num_labels = num_labels
        self.output_encoded_layers   = (args.output_encoded_layers=='true')
        self.output_attention_layers = (args.output_attention_layers=='true')
        self.bert = BertModel(config, output_score=(args.output_att_score=='true'), output_sum=(args.output_att_sum=='true'))
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        SPC任务分类
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param labels:
        :return:
        """
        sequence_output, pooled_output, attention_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                                    output_all_encoded_layers=(self.output_encoded_layers),
                                                                    output_all_attention_layers=(self.output_attention_layers))
        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels
        #assert len(sequence_output)==self.bert.config.num_hidden_layers + 1  # embeddings + 12 hiddens
        #assert len(attention_output)==self.bert.config.num_hidden_layers + 1 # None + 12 attentions
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits,labels)
            return logits, sequence_output, attention_output, loss
        else:
            return logits

def BertForGLUESimpleAdaptor(batch, model_outputs, no_logits, no_mask):
    dict_obj = {'hidden': model_outputs[1], 'attention': model_outputs[2]}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch[1]
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[0],)
    return dict_obj

def BertForGLUESimpleAdaptorTraining(batch, model_outputs):
    """
    返回模型的损失
    :param batch:
    :param model_outputs:
    :return:
    """
    return {'losses':(model_outputs[-1],)}


class ElectraClassificationHead(nn.Module):
    """CLS分类"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraSPC(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = True
        self.output_hidden_states = True
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()


    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        #形状 torch.Size([batch_size, seq_lenth, hidden_size])
        sequence_output = discriminator_hidden_states[0]
        #形状 torch.Size([batch_size, num_labels])
        logits = self.classifier(sequence_output)
        # 如果不为True，那么不能输出 self.output_attentions = True， self.output_hidden_states = True
        hidden_states, attention_output = discriminator_hidden_states[1:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, sequence_output, attention_output, loss
        else:
            return logits



class AlbertSPC(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = True
        self.output_hidden_states = True
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #形状 torch.Size([batch_size, seq_lenth, hidden_size])
        # sequence_output = outputs[0]
        #形状 torch.Size([batch_size, num_labels])
        # 如果不为True，那么不能输出 self.output_attentions = True， self.output_hidden_states = True
        hidden_states, attention_output = outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, pooled_output, attention_output, loss
        else:
            return logits