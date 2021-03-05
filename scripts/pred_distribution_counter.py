import json
from collections import defaultdict

with open("toy_data/training_data.txt", "r") as in_file, open("toy_data/pred_distribution.txt", "w") as out_file:
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
        # out_file.write(f"{key}: {pred_counter[key]}\n")
    predicate_list.sort(reverse=True, key=lambda x: x[1])
    for el in predicate_list:
        out_file.write(f"{el[0]}: {el[1]}\n")
    # out_file.write(json.dumps(reversed_pred_counter))


# Reverse dict so that (pred -> count) is instead (count -> pred)
