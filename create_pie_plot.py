import json
import matplotlib.pyplot as plt
from collections import defaultdict


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct


file_to_read_from = "toy_data/training_data.txt"

with open(file_to_read_from, "r") as in_file:
    pred_counter = defaultdict(int)

    for line in in_file:
        row = json.loads(line)
        if 'fact1_pred' in row:
            for pred in row['fact1_pred']:
                pred_counter[pred] += 1
        if 'fact2_pred' in row:
            for pred in row['fact2_pred']:
                pred_counter[pred] += 1

    predicate_list = list()
    for key in pred_counter.keys():
        predicate_list.append((key, pred_counter[key]))
    predicate_list.sort(reverse=True, key=lambda x: x[1])
predicates = [el[0] for el in predicate_list]
counts = [int(el[1]) for el in predicate_list]

# two_count = 0
# one_count = 0
# pop_indices = []
# for i, count in enumerate(counts):
#     if count == 2:
#         two_count += 1
#         pop_indices.append(i)
#     if count == 1:
#         one_count += 1
#         pop_indices.append(i)
# for idx in reversed(pop_indices):
#     counts.pop(idx)
#     predicates.pop(idx)
# print(one_count)
# print(two_count)
# counts = [count for count in counts if count != 1 and count != 2]
# counts.append(two_count)
# counts.append(one_count)
# counts = list(set(counts))
count_dict = defaultdict(int)
for count in counts:
    count_dict[count] += 1
# counts.sort()
plot_counts = list()
plot_labels = list()
for key in count_dict.keys():
    plot_counts.append(key)
    plot_labels.append(f"{count_dict[key]}")
# labels = [f"{count}" for count in counts]

# predicates.append("predicates with count of two")
# predicates.append("predicates with count of one")

plt.pie(plot_counts, labels=plot_labels)
plt.title('Predicate Distribution')
plt.axis('equal')
plt.show()

plt.savefig("pred_dist_pie.png")
