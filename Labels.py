import numpy as np
from sklearn.preprocessing import LabelBinarizer

topics = ['satire', 'opinion', 'reporting']

frames = [
    'Economic', 'Capacity_and_resources', 'Morality', 'Fairness_and_equality',
    'Legality_Constitutionality_and_jurisprudence', 'Policy_prescription_and_evaluation', 'Crime_and_punishment',
    'Security_and_defense', 'Health_and_safety', 'Quality_of_life', 'Cultural_identity', 'Public_opinion', 'Political',
    'External_regulation_and_reputation'
]

persuasion = [
    'Appeal_to_Authority', 'Appeal_to_Popularity', 'Appeal_to_Values', 'Appeal_to_Fear-Prejudice', 'Flag_Waving',
    'Causal_Oversimplification', 'False_Dilemma-No_Choice', 'Consequential_Oversimplification', 'Straw_Man',
    'Red_Herring', 'Whataboutism', 'Slogans', 'Appeal_to_Time', 'Conversation_Killer', 'Loaded_Language', 'Repetition',
    'Exaggeration-Minimisation', 'Obfuscation-Vagueness-Confusion', 'Name_Calling-Labeling', 'Doubt',
    'Guilt_by_Association', 'Appeal_to_Hypocrisy', 'Questioning_the_Reputation',
]

all_labels = topics + frames + persuasion
label_map = {all_labels[i]: i for i in range(len(all_labels))}
inverse_label_map = {i: all_labels[i] for i in range(len(all_labels))}
label_map['nothing'] = -1
inverse_label_map['-1'] = ''
_topic_label_encoder = LabelBinarizer()
_topic_label_encoder.fit(list(range(len(topics))))

_frame_label_encoder = LabelBinarizer()
_frame_label_encoder.fit(list(range(len(topics), len(topics) + len(frames))))

_persuasion_label_encoder = LabelBinarizer()
_persuasion_label_encoder.fit(list(range(len(topics) + len(frames), len(topics) + len(frames) + len(persuasion))))

encoders = {
    1: _topic_label_encoder,
    2: _frame_label_encoder,
    3: _persuasion_label_encoder
}


def label_encode(l, t_num):
    if l.__class__ == str or l.__class__ == int:
        l = [l]

    def encode_x(x):
        x_labels = x.replace(' ', '').split(',')
        return [label_map[i] for i in x_labels]

    encoders = {
        1: _topic_label_encoder,
        2: _frame_label_encoder,
        3: _persuasion_label_encoder
    }
    encoder = encoders[t_num]

    return list(map(lambda x: encoder.transform(encode_x(x)).sum(axis=0), l))


def label_decode(l, t_num):
    encoder = encoders[t_num]
    return np.array(list(map(lambda x: inverse_label_map[x], encoder.inverse_transform(l))))
