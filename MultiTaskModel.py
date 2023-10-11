from abc import ABC
from typing import Optional, List

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModel


class MultiTaskClassifierConfig(PretrainedConfig):
    model_type = 'MultiTaskClassifier'

    def __init__(
            self,
            task_nums: int = None,
            transformer_checkpoint: str = None,
            transformer_hidden_state_size: int = None,
            final_layer_size: int = 32,
            second_transformer_checkpoint: str = None,
            second_transformer_hidden_state_size: int = None,
            use_auto_encoder: bool = False,
            classes: List[List[str]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.task_nums = task_nums
        self.transformer_checkpoint = transformer_checkpoint
        self.second_transformer_checkpoint = second_transformer_checkpoint
        self.t1_last_layer_size = transformer_hidden_state_size
        self.t2_last_layer_size = second_transformer_hidden_state_size
        self.final_layer_size = final_layer_size
        self.classes = classes
        self.use_auto_encoder = use_auto_encoder


class MultiTaskClassifier(PreTrainedModel, ABC):
    config_class = MultiTaskClassifierConfig
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config: MultiTaskClassifierConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.t1 = AutoModel.from_pretrained(config.transformer_checkpoint, add_pooling_layer=True)
        self.use_auto_encoder = config.use_auto_encoder
        self.use_second_transformer = config.second_transformer_checkpoint is not None

        if self.use_second_transformer:
            self.t2 = AutoModel.from_pretrained(config.second_transformer_checkpoint, add_pooling_layer=True)

        if self.use_auto_encoder:
            self.t1_encoder, self.t1_decoder, = self.get_auto_encoder(
                config.t1_last_layer_size,
                config.final_layer_size
            )
            if self.use_second_transformer:
                self.t2_encoder, self.t2_decoder, = self.get_auto_encoder(
                    config.t2_last_layer_size,
                    config.final_layer_size
                )

        self.heads = []
        for i in range(len(config.classes)):
            self.heads.append(self._create_output_head(len(config.classes[i])))

        if self.use_second_transformer:
            if self.use_auto_encoder:
                classifiers_first_layer_input_size = 2 * config.final_layer_size
            else:
                classifiers_first_layer_input_size = config.t1_last_layer_size + config.t2_last_layer_size
        else:
            if self.use_auto_encoder:
                classifiers_first_layer_input_size = config.final_layer_size
            else:
                classifiers_first_layer_input_size = config.t1_last_layer_size

        self.hidden_state_to_head = nn.Sequential(
            nn.Linear(classifiers_first_layer_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(
            self,
            input_ids_t1: Optional[torch.LongTensor] = None,
            attention_mask_t1: Optional[torch.FloatTensor] = None,
            input_ids_t2: Optional[torch.LongTensor] = None,
            attention_mask_t2: Optional[torch.FloatTensor] = None,
    ):
        x = self.t1(input_ids_t1, attention_mask=attention_mask_t1)
        x = x.pooler_output
        if self.use_auto_encoder:
            e1 = self.t1_encoder(x)
            d1 = self.t1_decoder(e1)
        if self.use_second_transformer:
            y = self.t2(input_ids_t2, attention_mask=attention_mask_t2)
            y = y.pooler_output
            if self.use_auto_encoder:
                e2 = self.t2_encoder(y)
                d2 = self.t2_decoder(e2)

        if self.use_second_transformer and self.use_auto_encoder:
            hidden_state_to_classifier_input = torch.cat([e1, e2], dim=1)
        elif self.use_second_transformer:
            hidden_state_to_classifier_input = torch.cat([x, y], dim=1)
        elif self.use_auto_encoder:
            hidden_state_to_classifier_input = e1
        else:
            hidden_state_to_classifier_input = x

        classifier_input = self.hidden_state_to_head(hidden_state_to_classifier_input)
        outputs = {}
        if self.use_auto_encoder:
            outputs['t1_pooler'] = x
            outputs['d1_output'] = d1
            if self.use_second_transformer:
                outputs['t2_pooler'] = y
                outputs['d2_output'] = d2

        for i, head in enumerate(self.heads):
            outputs[f'logits_{i}'] = head(classifier_input)
        return outputs

    @staticmethod
    def get_auto_encoder(input_dim: int, latent_size: int):
        activation = nn.ReLU()
        encoder = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(input_dim, (input_dim + latent_size) // 2),
            activation,
            nn.Linear((input_dim + latent_size) // 2, latent_size),
            activation
        )
        decoder = nn.Sequential(
            nn.Linear(latent_size, (input_dim + latent_size) // 2),
            activation,
            nn.Linear((input_dim + latent_size) // 2, input_dim),
        )
        return encoder, decoder

    @staticmethod
    def _create_output_head(num_label: int):
        if num_label == 2:
            num_label = 1
        return nn.Linear(64, num_label)
