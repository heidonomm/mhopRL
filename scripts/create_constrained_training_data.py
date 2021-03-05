import json

"""
Script to remove all questions that have a fact associated with that only has ('be') as its predicate 
(because count says theres 332 in total)
"""
with open("toy_data/training_data.txt", 'r') as in_file, open("toy_data/constrained_training_data.txt", "w") as out_file:
    for index, sample in enumerate(in_file):
        row = json.loads(sample)
        if 'fact1_pred' not in row or 'fact2_pred' not in row:
            print(index, row['question'])
            continue

        if ('fact1_pred' in row and len(row['fact1_pred']) == 1 and row['fact1_pred'][0] == "be") or \
                ('fact2_pred' in row and len(row['fact2_pred']) == 1 and row['fact2_pred'][0] == "be"):
            print(index, row['question'])
        else:
            if len(row['fact1_pred']) == 0 or len(row['fact2_pred']) == 0:
                print(index, row['question'])
            else:
                out_file.write(f"{sample}")
