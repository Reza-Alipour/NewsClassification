from dataclasses import dataclass, field

import evaluate
import torch
import yaml
from datasets import load_dataset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, get_scheduler, AutoTokenizer

from MultiTaskModel import MultiTaskClassifierConfig, MultiTaskClassifier


@dataclass
class CustomizedTrainArguments:
    lr: float = field(default=1e-5)
    epochs: int = field(default=3)
    batch_size: int = field(default=4)


@dataclass
class ConfigArguments:
    dataset_config: str = field(default='configs/datasets-config.yaml')
    hf_read_token: str = field(default=None)
    model_checkpoint: str = field(default=None)
    push_to_hub: bool = field(default=False)
    repo_id: str = field(default=None)
    hf_write_token: str = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)


@dataclass
class ModelArguments:
    transformer1: str = field(default='xlm-roberta-base')
    t1_last_layer_size: int = field(default=768)
    transformer2: str = field(default=None)
    t2_last_layer_size: int = field(default=None)
    use_autoencoder: bool = field(default=False)
    final_shared_layer_size: int = field(default=32)


def prepare_data(x, dataset_config):
    if dataset_config['loss'] == 'CE':
        return {'text': x['text'], 'label': dataset_config['label_to_id'][x['label']]}
    elif dataset_config['loss'] == 'BCE' and dataset_config['class_nums'] == 2:
        return {'text': x['text'], 'label': [x['label']]}
    elif dataset_config['loss'] == 'BCE':
        label = [0] * dataset_config['class_nums']
        for l in x['label']:
            label[dataset_config['label_to_id'][l]] = 1
        return {'text': x['text'], 'label': label}
    else:
        raise ValueError(f'Unknown dataset type: {dataset_config["type"]}')


def tokenize(x, tokenizer1, tokenizer2=None):
    input_ids_t2 = None
    attention_mask_t2 = None

    t1_output = tokenizer1(
        x['text'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids_t1 = t1_output['input_ids']
    attention_mask_t1 = t1_output['attention_mask']
    if tokenizer2 is not None:
        t2_output = tokenizer2(
            x['text'],
            padding=False,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        input_ids_t2 = t2_output['input_ids']
        attention_mask_t2 = t2_output['attention_mask']

    labels = x['label']
    if labels.__class__ == list:
        labels = torch.cat([t.unsqueeze(0) for t in labels], dim=0).transpose(0, 1)
    tokenized_inputs = {
        'input_ids_t1': input_ids_t1,
        'attention_mask_t1': attention_mask_t1,
    }
    if tokenizer2 is not None:
        tokenized_inputs.update({
            'input_ids_t2': input_ids_t2,
            'attention_mask_t2': attention_mask_t2
        })
    return tokenized_inputs, labels


def main():
    parser = HfArgumentParser((ConfigArguments, ModelArguments, CustomizedTrainArguments))
    config_args, model_args, train_args = parser.parse_args_into_dataclasses()
    dataset_config = yaml.load(open(config_args.dataset_config, 'r'), Loader=yaml.FullLoader)
    datasets = [load_dataset(ds['name'], token=config_args.hf_read_token) for ds in dataset_config]
    datasets = [ds.filter(lambda x: x['text'] is not None and 11 > len(x['text'].split()) > 3) for ds in datasets]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer1 = AutoTokenizer.from_pretrained(model_args.transformer1)
    tokenizer2 = None if model_args.transformer2 is None else AutoTokenizer.from_pretrained(model_args.transformer2)
    if config_args.model_checkpoint:
        model_config = MultiTaskClassifierConfig.from_pretrained(
            config_args.model_checkpoint,
            token=config_args.hf_read_token
        )
        model = MultiTaskClassifier.from_pretrained(
            config_args.model_checkpoint,
            config=model_config,
            token=config_args.hf_read_token
        )
    else:
        model_config = MultiTaskClassifierConfig(
            task_nums=len(dataset_config),
            transformer_checkpoint=model_args.transformer1,
            transformer_hidden_state_size=model_args.t1_last_layer_size,
            second_transformer_checkpoint=model_args.transformer2,
            second_transformer_hidden_state_size=model_args.t2_last_layer_size,
            use_auto_encoder=model_args.use_autoencoder,
            final_layer_size=model_args.final_shared_layer_size,
            classes=[list(ds['label_to_id'].keys()) for ds in dataset_config]
        )
        model = MultiTaskClassifier(config=model_config)

    new_datasets = []
    valid_datasets = []
    for ds, ds_config in zip(datasets, dataset_config):
        train_ds = ds['train']
        valid_ds = ds['validation']
        if 'labels' in list(train_ds.features.keys()):
            train_ds = train_ds.rename_column('labels', 'label')
            valid_ds = valid_ds.rename_column('labels', 'label')
        if 'DoesUseSarcasm' in list(train_ds.features.keys()):
            train_ds = train_ds.rename_column('DoesUseSarcasm', 'label')
            valid_ds = valid_ds.rename_column('DoesUseSarcasm', 'label')
        if 'IsPositive' in list(train_ds.features.keys()):
            train_ds = train_ds.rename_column('IsPositive', 'label')
            valid_ds = valid_ds.rename_column('IsPositive', 'label')
        column_to_remove = list(train_ds.features.keys())
        column_to_remove.remove('text')
        column_to_remove.remove('label')
        new_datasets.append(train_ds.map(lambda x: prepare_data(x, ds_config), remove_columns=column_to_remove))
        valid_datasets.append(valid_ds.map(lambda x: prepare_data(x, ds_config), remove_columns=column_to_remove))
    datasets = new_datasets

    dataloaders = [DataLoader(ds, batch_size=train_args.batch_size, shuffle=True) for ds in datasets]
    validation_dataloaders = [DataLoader(ds, batch_size=train_args.batch_size, shuffle=False) for ds in valid_datasets]
    freqs = [ds['freqs'] for ds in dataset_config]

    loss_functions = []
    for i, ds in enumerate(dataset_config):
        if ds['loss'] == 'CE':
            loss_functions.append(CrossEntropyLoss())
        elif ds['loss'] == 'BCE':
            loss_functions.append(BCEWithLogitsLoss())
        else:
            raise NotImplementedError(f'{ds["loss"]} loss function is not supported yet.')
    model.to(device)

    iterators = [iter(dl) for dl in dataloaders]
    optimizer = AdamW(model.parameters(), lr=train_args.lr)
    num_training_steps = train_args.epochs * sum([len(dl) for dl in dataloaders])
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    metric = evaluate.load("accuracy")
    for e in range(train_args.epochs):
        epoch_completed = [False] * len(dataset_config)
        model.train()
        while config_args.do_train and not all(epoch_completed):
            for f_ind, f in enumerate(freqs):
                for _ in range(f):
                    try:
                        batch = next(iterators[f_ind])
                    except StopIteration:
                        epoch_completed[f_ind] = True
                        iterators[f_ind] = iter(dataloaders[f_ind])
                        continue

                    input_batch, labels = tokenize(batch, tokenizer1, tokenizer2)
                    input_batch = {k: v.to(device) for k, v in input_batch.items()}
                    labels = labels.to(device)

                    outputs = model(**input_batch)
                    logits = outputs[f'logits_{f_ind}']
                    loss_fct = loss_functions[f_ind]
                    if loss_fct.__class__ is torch.nn.modules.loss.BCEWithLogitsLoss:
                        labels = labels.float()
                    loss = loss_fct(logits, labels)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        if config_args.do_eval and e % 4 == 0:
            model.eval()
            for i in range(len(valid_datasets)):
                eval_dataloader = validation_dataloaders[i]
                for batch in eval_dataloader:
                    input_batch, labels = tokenize(batch, tokenizer1, tokenizer2)
                    input_batch = {k: v.to(device) for k, v in input_batch.items()}
                    labels = labels.to(device)
                    with torch.no_grad():
                        outputs = model(**input_batch)
                    logits = outputs[f'logits_{i}']
                    loss_type = dataset_config[i]['loss']
                    if loss_type == 'CE':
                        predicted_classes = torch.argmax(logits, dim=1)
                        metric.add_batch(references=labels, predictions=predicted_classes)
                    elif loss_type == 'BCE':
                        threshold = 0.0  # Todo: Use validation set to find the best threshold
                        predicted_classes = (logits > threshold).long()
                        predicted_classes = predicted_classes.view(-1)
                        labels = labels.reshape(-1).long()
                        metric.add_batch(references=labels, predictions=predicted_classes)

                print(f'Epoch: {e}, Dataset: {dataset_config[i]["name"]}, Accuracy: {metric.compute()["accuracy"]}.')
            if config_args.do_train is False:
                break

    if config_args.push_to_hub:
        model.push_to_hub(
            repo_id=config_args.repo_id,
            token=config_args.hf_write_token
        )


if __name__ == '__main__':
    main()
