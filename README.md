# A Multilingual Transformer Language Model-based Approach for the Detection of News Genre, Framing and Persuasion Techniques

This project is a submission for SemEval contest that involves three sub-tasks: news genre classification, news framing
and persuasion technique identification. We aim to build a model that can accurately perform these tasks.

## Task Description

- **News Genre Classification**: Given a news article, the model needs to classify it into one of three genres: Sports,
  Politics, and Entertainment.
- **News Framing**: Given a news article, the model needs to identify its framing by assigning it one of the following
  labels: Conflict, Human Interest, and Economic.
- **Persuasion Technique Identification**: Given a news article, the model needs to identify the persuasion techniques
  used by the reporter.

## Approach

Our project aims to find solutions for the three subtasks by utilizing two pre-trained transformer models - XLM and
LaBSE - and ensembling them.Our model is designed to perform all three tasks simultaneously, making it a multi-task
model. To facilitate this, we use separate "heads" - specialized layers - for each subtask. To enable information
sharing across the subtasks and make the best use of our pre-trained models, we make the final layer of the
transformers trainable. This enables us to create common layers shared by all subtasks. Finally, we incorporate data
augmentation techniques to enhance the model's performance.


