import json
import requests
import nltk
from nltk import word_tokenize, WordNetLemmatizer, pos_tag


filler_verbs = ["be", "is", "are", "did", "do", "have", "had", "can"]
out_preds = set()
error_facts = set()


def preprocess(text):
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]

    tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
    return " ".join(tokens)


def remove_filler_verb(sentence):
    for index, pair in enumerate(sentence):
        if pair[0] in filler_verbs:
            sentence.pop(index)
    return sentence


def preprocess_verbs(list_of_verbs):
    lemma = WordNetLemmatizer()
    verbs = [lemma.lemmatize(word.lower(), pos="v") for word in list_of_verbs]
    return verbs


def get_predicate_from_tags(tagged_sentence):
    pred = ""
    for i, tag in enumerate(tagged_sentence):
        if i < len(tagged_sentence) - 1:
            pred += f"{tag[0]} "
        else:
            pred += tag[0]
    return pred


def multiple_verbs(tagged_sentence):
    verb_seen = False
    for tag in tagged_sentence:
        if tag[1] == "VERB" and not verb_seen:
            verb_seen = True
        elif tag[1] == "VERB" and verb_seen:
            return True
    return False


def remove_empty_verbs(predicate):
    tokens = word_tokenize(predicate)
    tags = pos_tag(tokens, tagset="universal")

    if multiple_verbs(tags):
        return_sent = get_predicate_from_tags(
            remove_filler_verb(tags)).rstrip()
        out_preds.add(return_sent)
    else:
        return_sent = predicate
    return return_sent


def get_verbs(sent):
    tagged = pos_tag(word_tokenize(sent), tagset='universal')
    verbs = set()
    for tag in tagged:
        if tag[1] == "VERB":
            verbs.add(tag[0])
    if len(verbs) == 0:
        return []
    lemmad_verbs = preprocess_verbs(list(verbs))
    filtered_verbs = [
        verb for verb in lemmad_verbs if verb not in filler_verbs]
    return filtered_verbs


with open("data/QASC_Dataset/dev.jsonl", "r") as in_file, open("toy_data/verb_only/training_data.txt", "w") as out_file, \
        open("erroneous_facts.txt", "w") as error_out:
    for i, line in enumerate(in_file):
        data = {}
        print(i)
        json_line = json.loads(line)
        data["question"] = preprocess(json_line["question"]['stem'])

        for index, choice in enumerate(json_line['question']['choices']):
            json_line['question']['choices'][index]['text'] = preprocess(
                choice['text'])

        data['choices'] = json_line['question']['choices']
        fact1_verbs = get_verbs(json_line['fact1'])
        fact2_verbs = get_verbs(json_line['fact2'])
        if len(fact1_verbs) == 0 or len(fact2_verbs) == 0:
            continue

        data['formatted_question'] = preprocess(
            json_line['formatted_question'])
        data['answer'] = json_line['answerKey']
        data['fact1'] = preprocess(json_line['fact1'])
        data['fact2'] = preprocess(json_line['fact2'])
        # data['fact1_pred'] = list()
        data['fact1_pred'] = fact1_verbs
        data['fact2_pred'] = fact2_verbs

        out_file.write(f"{json.dumps(data)}\n")
        # if i >= 1:
        #     break
    for fact in error_facts:
        error_out.write(f'{fact}\n')

with open("all_predicates.txt", "w") as predicate_file:
    for predicate in out_preds:
        predicate_file.write(f"{predicate}\n")
