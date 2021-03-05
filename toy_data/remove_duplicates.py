import random

with open("toy_data/dev_norm_predicates.txt", "r") as in_file, open("toy_data/dev_norm_unique_predicates.txt", "w") as out_file:
    unique_predicates = set()
    for line in in_file:
        unique_predicates.add(line)

    predicates_list = list()
    for predicate in unique_predicates:
        predicates_list.append(predicate)
    random.shuffle(predicates_list)
    for predicate in predicates_list:
        out_file.write(predicate)
