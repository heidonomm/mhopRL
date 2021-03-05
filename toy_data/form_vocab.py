import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer

import json
import string


# q_string = '{"id": "3NGI5ARFTT4HNGVWXAMLNBMFA0U1PG", "question": {"stem": "Climate is generally described in terms of what?", "choices": [{"text": "sand", "label": "A"}, {"text": "occurs over a wide range", "label": "B"}, {"text": "forests", "label": "C"}, {"text": "Global warming", "label": "D"}, {"text": "rapid changes occur", "label": "E"}, {"text": "local weather conditions", "label": "F"}, {"text": "measure of motion", "label": "G"}, {"text": "city life", "label": "H"}]}, "answerKey": "F", "fact1": "Climate is generally described in terms of temperature and moisture.", "fact2": "Fire behavior is driven by local weather conditions such as winds, temperature and moisture.", "combinedfact": "Climate is generally described in terms of local weather conditions", "formatted_question": "Climate is generally described in terms of what? (A) sand (B) occurs over a wide range (C) forests (D) Global warming (E) rapid changes occur (F) local weather conditions (G) measure of motion (H) city life"}'
# q_json = json.loads(q_string)


# def state_rep_generator(q_obj):
#     state = ""
#     elements = list()
#     elements.append(preprocess(q_obj["stem"]))
#     for choice in q_obj['choices']:
#         print(choice['text'])
#         elements.append((preprocess(choice['text'])))
#     print(elements)
#     return "<|>".join(elements)


def preprocess(text):
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]

    tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
    return tokens


# print(state_rep_generator(q_json['question']))


vocab_set = set()
for letter in string.ascii_lowercase:
    vocab_set.add(letter)


# with open("data/QASC_Dataset/dev.jsonl", "r") as in_dev_file, open("toy_data/dev_norm_unique_predicates.txt", "r") as in_pred_file, open("word_vocab.txt", "w") as out_file:
#     for line in in_dev_file:
#         line = json.loads(line)
#         for stem_word in preprocess(line["question"]["stem"]):
#             vocab_set.add(stem_word)
#         # vocab_set.add(word for word in preprocess(line["question"]["stem"]))
#         # choices = [preprocess(choice["text"]) for choice in line["question"]["choices"]]
#         for choice in line["question"]["choices"]:
#             for choice_word in preprocess(choice["text"]):
#                 vocab_set.add(choice_word)
#             # vocab_set.add(word for word in preprocess(choice["text"]))

#     for index, line in enumerate(in_pred_file):
#         for word in preprocess(line):
#             vocab_set.add(word)
#         # vocab_set.add(word for word in preprocess(line))

#     for word in vocab_set:
#         # print(word)
#         out_file.write(f"{word}\n")
