import torch
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, \
    get_linear_schedule_with_warmup

from HelperClasses import CustomTrainer, LogCallback
from HelperFunctions import convert_to_features, compute_metrics_f1_multi, compute_metrics
from Models import MultiHeadClassifier, MultiHeadModel
from Parameters import xlm_checkpoint, labse_checkpoint, t2_weights, total_epochs, batch_size, each_task_epochs, \
    t3_weights, t1_weights

xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_checkpoint)
labse_tokenizer = AutoTokenizer.from_pretrained(labse_checkpoint)

AutoConfig.register("MultiHeadClassifier", MultiHeadClassifier)
AutoModel.register(MultiHeadClassifier, MultiHeadModel)

loaded_csvs = {
    (1, 'train1'): Dataset.from_csv(f'dataset/t1/train.csv'),
    (2, 'train2'): Dataset.from_csv(f'dataset/t2/train.csv'),
    (3, 'train3'): Dataset.from_csv(f'dataset/t3/train.csv'),
    (1, 'validation1'): Dataset.from_csv(f'dataset/t1/validation.csv'),
    (2, 'validation2'): Dataset.from_csv(f'dataset/t2/validation.csv'),
    (3, 'validation3'): Dataset.from_csv(f'dataset/t3/validation.csv'),
}
dataset = {}
for (i, j), k in loaded_csvs.items():
    tokenized = k.map(
        lambda x: convert_to_features(x, i, xlm_tokenizer, labse_tokenizer),
        batched=True,
        batch_size=16
    )
    dataset[j] = tokenized
tokenized_dataset = DatasetDict(dataset)

model = MultiHeadModel(config=AutoConfig.from_pretrained('results'))
# model = MultiHeadModel.from_pretrained('model_results')

train_task = 1
for i in range(total_epochs):
    args = TrainingArguments(
        "results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=each_task_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        save_total_limit=4,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro',
        gradient_accumulation_steps=16,
        do_train=True,
        do_eval=True,
        report_to='none',
        per_device_eval_batch_size=batch_size,
        logging_steps=1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = int((len(loaded_csvs['train']) // batch_size) * each_task_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    metric = None
    if train_task == 1:
        metric = lambda x: compute_metrics(x, train_task)
    else:
        metric = lambda x: compute_metrics_f1_multi(x, train_task)
    loss = 'ce' if train_task == 1 else 'bce'
    num_labels = 3 if train_task == 1 else (14 if train_task == 2 else 23)
    loss_weights = t1_weights if train_task == 1 else (t2_weights if train_task == 2 else t3_weights)
    trainer = CustomTrainer(
        model,
        args,
        optimizers=(optimizer, lr_scheduler),
        train_dataset=tokenized_dataset[f'train{train_task}'],
        eval_dataset=tokenized_dataset[f'validation{train_task}'],
        compute_metrics=metric,
        callbacks=[LogCallback],
        task=train_task,
        num_labels=num_labels,
        loss=loss,
        weights=loss_weights,
    )
    trainer.train()

    train_task += 1

model.save_pretrained("model_results")
