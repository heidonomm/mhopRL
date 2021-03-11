import json
from collections import defaultdict


# file_to_read_from = "toy_data/verb_only/training_data.txt"
# file_to_write_to = "toy_data/verb_only/constrained_training_data.txt"

"""
    get counts of predicates based on previous constrained dataset
"""
with open("toy_data/constrained_training_data.txt", "r") as in_file:
    pred_counter = defaultdict(int)

    for line in in_file:
        row = json.loads(line)
        for pred in row['fact1_pred']:
            pred_counter[pred] += 1
        for pred in row['fact2_pred']:
            pred_counter[pred] += 1


"""
    Script to remove all questions that have a fact associated with it
    that only has ('be') as its predicate (because count says theres 332 in total)
    also questions which have a fact which has a predicate which occurs less than X times
"""
threshold = 3
with open(file_to_read_from, 'r') as in_file, open(file_to_write_to, "w") as out_file:
    for index, sample in enumerate(in_file):
        row = json.loads(sample)
        constrained_predicates = list()
        for pred in row['fact1_pred']:
            if pred_counter[pred] >= threshold:
                constrained_predicates.append(pred)
        if len(constrained_predicates) == 0:
            continue
        else:
            row['fact1_pred'] = constrained_predicates
        constrained_predicates = list()
        for pred in row['fact2_pred']:
            if pred_counter[pred] >= threshold:
                constrained_predicates.append(pred)
        if len(constrained_predicates) == 0:
            continue
        else:
            row['fact2_pred'] = constrained_predicates
        # if it reaches this stage, then all predicates for current sample have atleast count of 5
        out_file.write(f"{json.dumps(row)}\n")
