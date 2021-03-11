import json

# file_to_read_from = "toy_data/verb_only/constrained_training_data.txt"
# file_to_write_to = "toy_data/verb_only/constrained_predicate_list.txt"

all_preds = set()
with open(file_to_read_from, 'r') as in_file, open(file_to_write_to, "w") as out_file:
    for sample in in_file:
        row = json.loads(sample)
        for pred in row['fact1_pred']:
            all_preds.add(pred)
        for pred in row['fact2_pred']:
            all_preds.add(pred)

    preds_sorted = list(all_preds)
    preds_sorted.sort()
    for pred in preds_sorted:
        out_file.write(f"{pred.strip()}\n")
