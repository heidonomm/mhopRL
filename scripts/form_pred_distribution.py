import json
from collections import defaultdict

# file_to_read_from = "toy_data/verb_only/training_data.txt"
# file_to_write_to = "toy_data/verb_only/pred_distribution.txt"

with open(file_to_read_from, "r") as in_file, open(file_to_write_to, "w") as out_file:
    pred_counter = defaultdict(int)

    for line in in_file:
        row = json.loads(line)
        if 'fact1_pred' in row:
            for pred in row['fact1_pred']:
                pred_counter[pred] += 1
        if 'fact2_pred' in row:
            for pred in row['fact2_pred']:
                pred_counter[pred] += 1

    reversed_pred_counter = {count: pred for (
        pred, count) in pred_counter.items()}

    predicate_list = list()
    for key in pred_counter.keys():
        predicate_list.append((key, pred_counter[key]))
    predicate_list.sort(reverse=True, key=lambda x: x[1])
    for el in predicate_list:
        out_file.write(f"{el[0]}: {el[1]}\n")
