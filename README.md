# Multi-Task News Analysis with Transformers

This project is our submission for the SemEval contest, encompassing three distinct sub-tasks: news genre
classification, news framing detection, and persuasion technique identification.

## Task Description

- **News Genre Classification**: Given a news article, the model needs to classify it into one of three genres: Satire,
  Opinion, and Reporting.
- **News Framing**: Given a news article, the model needs to identify its framing by assigning it one of the following
  labels:<br> Economic, Capacity_and_resources, Morality, Fairness_and_equality,
  Legality_Constitutionality_and_jurisprudence, Policy_prescription_and_evaluation, Crime_and_punishment,
  Security_and_defense, Health_and_safety, Quality_of_life, Cultural_identity, Public_opinion, Political,
  and External_regulation_and_reputation.
- **Persuasion Technique Identification**: Given a news article, the model needs to identify the persuasion techniques
  used by the reporter. The techniques are:<br>
  Appeal_to_Authority, Appeal_to_Popularity, Appeal_to_Values, Appeal_to_Fear-Prejudice, Flag_Waving,
  Causal_Oversimplification, False_Dilemma-No_Choice, Consequential_Oversimplification, Straw_Man, Red_Herring
  ,Whataboutism ,Slogans, Appeal_to_Time, Conversation_Killer, Loaded_Language, Repetition, Exaggeration-Minimisation,
  Obfuscation-Vagueness-Confusion, Name_Calling-Labeling, Doubt, Guilt_by_Association, Appeal_to_Hypocrisy,
  Questioning_the_Reputation,### Note

_Please note that the dataset used for this project is not publicly available._

## Approach

Our project addresses three distinct sub-tasks: news genre classification, news framing detection, and persuasion
technique identification. We aim to develop a multi-task model using state-of-the-art Transformer architectures that
excel in these challenging tasks.

#### _Features_:

- **Modular Classification**: We use individual classification heads for each task, allowing independent fine-tuning for
  optimal performance.
- **Ensembling Transformers**: Our model can seamlessly incorporate up to two Transformer models, leveraging ensemble
  techniques to enhance predictive capabilities.
- **Autoencoder Integration**: An optional autoencoder can refine features by learning representations, improving
  classification quality and avoiding overfitting.
- **Adaptability**: During training, you can choose settings like using a second Transformer or enabling the
  autoencoder, tailoring the model to specific task requirements.

## Training

### Installing Dependencies

```pip install -r requirements.txt```

### Training the Model

```
python Main.py --lr \
 learning_rate (default = 1e-5) \
  --epochs num_epochs (default = 3) \
  --batch_size batch_size (default = 4) \
  --dataset_config path/to/dataset_config.yaml (default = configs/datasets-config.yaml) \
  --hf_read_token token_to_read_from_huggingface (default = None) \
  --model_checkpoint trained_model_checkpoint (default = None) \
  --transformer1 transformer_name (default = 'xlm-roberta-base') \
  --t1_last_layer_size last_layer_size (default = 768) \
  --transformer2 transformer_name (default = None) \
  --t2_last_layer_size last_layer_size (default = None) \
  --use_autoencoder (default = False) \
  --final_shared_layer_size final_shared_layer_between_tasks_heads_size (default = 32) \
  --push_to_hub (default = False) \
  --repo_id repo_id_to_push (default = None) \
  --hf_write_token token_to_write_on_huggingface (default = None) \
  --do_train (default = False) \
  --do_eval (default = False)
```

### Using the Model for Custom Tasks

Easily adapt this model for your specific NLP tasks by customizing the dataset configuration in
the `configs/datasets_config.yaml` file. Train the model with your data, and you're ready to make predictions tailored
to your needs.



