import matplotlib.pyplot as plt


# file_to_read_from1 = "toy_data/verb_only/accuracy.txt"
# file_to_read_from2 = "toy_data/verb_only/loss.txt"

# experiment_name = 'Verb only; Actions: 277; Samples: 458, Epochs: 30'

# file_to_read_from1 = "constrained_acc.txt"
# file_to_read_from2 = "constrained_loss.txt"

# experiment_name = 'Multiword verb, Actions: 82; Samples: 127, Episodes: 30'

with open(file_to_read_from1, 'r') as acc_file, open(file_to_read_from2, "r") as loss_file:
    accuracies = acc_file.read().splitlines()
    loss = loss_file.read().splitlines()

accuracies = [round(float(acc), 2) for acc in accuracies]
loss = [round(float(lss), 2) for lss in loss]
plt.plot(accuracies)
plt.plot(loss)
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.xlabel('epochs')
plt.title(experiment_name)
plt.savefig("plots/" + experiment_name + ".png")
plt.show()
