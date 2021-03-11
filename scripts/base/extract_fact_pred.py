import json
import requests
import nltk
from nltk import word_tokenize, WordNetLemmatizer, pos_tag


filler_verbs = ["be", "is", "are", "did", "do", "have", "had", "can"]
out_preds = set()
error_facts = set()


def call_stanford_openie(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response


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


with open("data/QASC_Dataset/dev.jsonl", "r") as in_file, open("toy_data/training_data.txt", "w") as out_file, \
        open("erroneous_facts.txt", "w") as error_out:
    for i, line in enumerate(in_file):
        data = {}
        print(i)
        json_line = json.loads(line)
        data["question"] = preprocess(json_line["question"]['stem'])

        fact1 = call_stanford_openie(json_line["fact1"])
        fact2 = call_stanford_openie(json_line["fact2"])

        # data['choices'] = preprocess(json_line['question']['choices'])
        for index, choice in enumerate(json_line['question']['choices']):
            json_line['question']['choices'][index]['text'] = preprocess(
                choice['text'])

        data['choices'] = json_line['question']['choices']

        data['formatted_question'] = preprocess(
            json_line['formatted_question'])
        data['answer'] = json_line['answerKey']
        data['fact1'] = preprocess(json_line['fact1'])
        data['fact2'] = preprocess(json_line['fact2'])
        # data['fact1_pred'] = list()
        data['fact2_pred'] = list()

        if len(fact1['sentences']) > 0:
            if len(fact1['sentences'][0]['openie']) > 0:
                sentence_predicates = list()
                for openie in fact1['sentences'][0]['openie']:
                    pred = preprocess(openie['relation'])
                    pred = remove_empty_verbs(pred)
                    sentence_predicates.append(pred)
                for index in range(len(sentence_predicates) - 1, -1, -1):
                    if len(sentence_predicates) > 1:
                        if sentence_predicates[index] in filler_verbs:
                            sentence_predicates.pop(index)
                data["fact1_pred"] = list(set(sentence_predicates))
                for pred in sentence_predicates:
                    out_preds.add(pred)
            else:
                error_facts.add(json_line['fact1'])
                # error_out.write(f"{json_line['fact1']}\n")
        else:
            # error_out.write(f"{json_line['fact2']}\n")
            error_facts.add(json_line['fact1'])

        if len(fact2['sentences']) > 0:
            if len(fact2['sentences'][0]['openie']) > 0:
                sentence_predicates = list()
                for openie in fact2['sentences'][0]['openie']:
                    pred = preprocess(openie['relation'])
                    pred = remove_empty_verbs(pred)
                    sentence_predicates.append(pred)
                for index in range(len(sentence_predicates) - 1, -1, -1):
                    if len(sentence_predicates) > 1:
                        if sentence_predicates[index] in filler_verbs:
                            sentence_predicates.pop(index)
                data['fact2_pred'] = list(set(sentence_predicates))
                for pred in sentence_predicates:
                    out_preds.add(pred)
            else:
                # error_out.write(f"{json_line['fact2']}\n")
                error_facts.add(json_line['fact2'])
        else:
            # error_out.write(f"{json_line['fact2']}\n")
            error_facts.add(json_line['fact2'])

        out_file.write(f"{json.dumps(data)}\n")
        # if i >= 1:
        #     break
    for fact in error_facts:
        error_out.write(f'{fact}\n')

with open("all_predicates.txt", "w") as predicate_file:
    for predicate in out_preds:
        predicate_file.write(f"{predicate}\n")
