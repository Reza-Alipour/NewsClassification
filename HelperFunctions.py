import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from Labels import label_encode
from Parameters import tokenizer_max_size


def compute_metrics(p, task_to_train):
    pred, labels = p
    pred = pred[task_to_train - 1]
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred, normalize=True)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    micro = f1_score(labels, pred, average="micro")
    macro = f1_score(labels, pred, average="macro")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "micro": micro, "macro": macro}


def compute_metrics_f1_multi(p, task_to_train):
    predictions, labels = p
    predictions = predictions[task_to_train - 1]
    prediction_s = torch.sigmoid(torch.tensor(predictions)).numpy()
    pred = np.zeros_like(prediction_s)
    pred[prediction_s > 0.5] = 1
    micro = f1_score(labels, pred, average="micro")
    macro = f1_score(labels, pred, average="macro")

    return {"micro": micro, "macro": macro}


def convert_to_features(samples, t_num, xlm_tokenizer, labse_tokenizer):
    xlm_output = xlm_tokenizer(
        samples['text'],
        max_length=tokenizer_max_size,
        padding='max_length',
        truncation=True
    )
    labse_output = labse_tokenizer(
        samples['text'],
        max_length=tokenizer_max_size,
        padding='max_length',
        truncation=True
    )
    labels = label_encode(samples['labels'], t_num)
    if t_num == 1:
        labels = np.argmax(labels, axis=1)
    return {
        'xlm_input_ids': xlm_output['input_ids'],
        'xlm_attention_mask': xlm_output['attention_mask'],
        'labse_input_ids': labse_output['input_ids'],
        'labse_attention_mask': labse_output['attention_mask'],
        'labels': labels,
    }
