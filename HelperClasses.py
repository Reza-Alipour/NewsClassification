from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import Trainer, BatchEncoding, TrainerCallback

from Parameters import device


class CustomTrainer(Trainer):
    def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
            task=1,
            num_labels=3,
            loss='ce',
            weights=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.task = task
        self.loss = loss
        self.weights = torch.tensor(weights).to(device)
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs[self.task]
        loss = None
        if labels is not None:
            if self.loss == 'ce':
                loss_fct = CrossEntropyLoss(weight=self.weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.loss == 'bce':
                loss_fct = BCEWithLogitsLoss(weight=self.weights)
                loss = loss_fct(logits, labels.double())
            else:
                raise NotImplementedError()

        return (loss, outputs) if return_outputs else loss


@dataclass
class SimpleDataCollator:
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = defaultdict(list)
        for feature in features:
            for k, v in feature.items():
                encoded_inputs[k].append(v)
        batch = BatchEncoding(encoded_inputs, tensor_type=self.return_tensors)
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


class LogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print('Callback log ------------------------------------------------------------')
        if state.log_history is None:
            print('Callback finished --------------------------------------------------------')
            return

        try:
            for i in state.log_history[-2:]:
                if i and i.__class__ == dict:
                    for k, v in i.items():
                        print(f'{k}: {v}')
                print('-----------------------------------------------------------------')
        except:
            print("Failed to print logs")

        print('Callback finished --------------------------------------------------------')
