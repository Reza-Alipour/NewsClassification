from typing import Optional

import torch
from torch import nn
from transformers import BertPreTrainedModel, XLMRobertaPreTrainedModel, \
    PretrainedConfig, XLMRobertaModel, BertModel, PreTrainedModel

from Parameters import device, xlm_checkpoint, labse_checkpoint


class XLMRoberta(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = XLMRobertaModel(config, add_pooling_layer=True)
        for name, param in self.roberta.named_parameters():
            if any([i in name for i in ['23', 'pooler']]):
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False
        self.post_init()


class Bert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        for name, param in self.bert.named_parameters():
            if any([i in name for i in ['11', 'pooler']]):
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False
        self.post_init()


class MultiHeadClassifier(PretrainedConfig):
    model_type = "MultiHeadClassifier"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t1_model = kwargs.pop('t1_model', 'simple')
        self.t2_model = kwargs.pop('t2_model', 'simple')
        self.t3_model = kwargs.pop('t3_model', 'simple')
        self.xlm_r = kwargs.pop('xlm_r', xlm_checkpoint)
        self.mbert = kwargs.pop('mbert', labse_checkpoint)


class MultiHeadModel(PreTrainedModel):
    config_class = MultiHeadClassifier
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: MultiHeadClassifier, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.xlm_r = XLMRoberta.from_pretrained(config.xlm_r).roberta
        self.labse = Bert.from_pretrained(config.mbert).bert
        self.head1 = self._create_output_head('simple', 3)
        self.head2 = self._create_output_head('simple', 14)
        self.head3 = self._create_output_head('simple', 23)

    def _create_output_head(self, model_name: str, num_label: int):
        if model_name == 'simple':
            return SimpleHead(num_label)
        else:
            raise NotImplementedError()

    def forward(
            self,
            xlm_input_ids: Optional[torch.LongTensor] = None,
            xlm_attention_mask: Optional[torch.FloatTensor] = None,
            labse_input_ids: Optional[torch.LongTensor] = None,
            labse_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ):
        xlm_output = self.xlm_r(
            xlm_input_ids,
            attention_mask=xlm_attention_mask,
        )
        bert_output = self.labse(
            labse_input_ids,
            attention_mask=labse_attention_mask
        )
        bert_cls = bert_output.last_hidden_state[:, 0, :]
        xlm_cls = xlm_output.last_hidden_state[:, 0, :]
        logits1 = self.head1(xlm_output=xlm_cls, bert_output=bert_cls)
        logits2 = self.head2(xlm_output=xlm_cls, bert_output=bert_cls)
        logits3 = self.head3(xlm_output=xlm_cls, bert_output=bert_cls)

        return None, logits1, logits2, logits3


class SimpleHead(nn.Module):
    def __init__(self, num_label: int):
        super().__init__()
        self.num_labels = num_label
        self.xlm_classifier = ClassificationHead(1024, num_label)
        self.labse_classifier = ClassificationHead(768, num_label)
        self.last_layer = nn.Linear(2 * num_label, num_label)
        self.weights = nn.Parameter(0.5 * torch.ones(self.num_labels).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.num_labels).to(device), requires_grad=True)

    def forward(self, xlm_output=None, bert_output=None):
        xlm_feature = xlm_output
        labse_feature = bert_output
        xlm_logits = self.xlm_classifier(xlm_feature)
        labse_logits = self.labse_classifier(labse_feature)
        logits = self.weights * xlm_logits + (1 - self.weights) * labse_logits
        logits += self.bias
        return logits


class ClassificationHead(nn.Module):

    def __init__(self, input_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.13)
        self.out_proj = nn.Linear(input_size, num_labels)
        self._init_weights()

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
