import json
from collections import defaultdict

import yaml
import torch
import torch.optim as optim
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN
from schedule import ExponentialSchedule
from replay import *
from utils import words_to_ids, to_one_hot


class DQNTrainer(object):
    def __init__(self):
        with open('config.yaml') as cf:
            self.config = yaml.safe_load(cf)

        self.vocab = self.load_vocab()
        self.all_actions = self.load_action_dictionary()
        self.training_dataset = self.load_training_dataset()

        self.word2index = self.load_word2index()
        self.action2index = self.load_action2index()
        self.action_idx2freq = self.load_action2freq()

        self.num_epochs = self.config['general']['max_epoch']
        self.num_episodes = self.config['general']['max_episode']
        self.hidden_size = self.config['general']['hidden_size']
        self.gamma = self.config['general']['gamma']

        self.action_threshold = self.config['general']['action_threshold']
        self.imbalance_ratio = self.get_imbalance_ratio()

        self.experiment_name = f'Constrained Verb, custom reward, Samples:{len(self.training_dataset)}, Actions:{len(self.all_actions)}, Epochs: {self.num_epochs}'

        self.model = DQN(len(self.vocab), len(
            self.all_actions), self.hidden_size)
        self.optimizer = optim.Adam(
            self.model.parameters(), self.config['optimizer']['learning_rate'])
        self.epsilon_scheduler = ExponentialSchedule(
            self.num_epochs * self.num_episodes, self.config['epsilon']['decay_rate'], self.config['epsilon']['final_value'])

        self.queue_size = 50
        self.loss_queue = []
        self.accuracy_queue = []

    def add_loss(self, loss_value):
        if len(self.loss_queue) > self.queue_size:
            self.loss_queue.pop(0)
            self.loss_queue.append(loss_value)
        else:
            self.loss_queue.append(loss_value)

    def add_accuracy(self, acc_value):
        if len(self.accuracy_queue) > self.queue_size:
            self.accuracy_queue.pop(0)
            self.accuracy_queue.append(acc_value)
        else:
            self.accuracy_queue.append(acc_value)

    def get_avg_loss_acc(self):
        avg_loss = sum(self.loss_queue) / len(self.loss_queue)
        avg_acc = sum(self.accuracy_queue) / len(self.accuracy_queue)
        return avg_loss, avg_acc

    def get_imbalance_ratio(self):
        majority_count = 0
        minority_count = 0
        for key in self.action_idx2freq.keys():
            if self.action_idx2freq[key] >= self.action_threshold:
                minority_count += self.action_idx2freq[key]
            else:
                majority_count += self.action_idx2freq[key]
        print(
            f"majority count is: {majority_count}, minority: {minority_count}")
        return minority_count / majority_count

    def load_action2index(self):
        action_dict = dict()
        for index, action in enumerate(self.all_actions):
            action_dict[action.rstrip()] = index
        return action_dict

    def load_word2index(self):
        index_dict = dict()
        for index, word in enumerate(self.vocab):
            index_dict[word.rstrip()] = index
        return index_dict

    def load_vocab(self):
        return open("toy_data/word_vocab.txt").read().splitlines()

    def load_action_dictionary(self):
        return open("toy_data/verb_only/constrained_predicate_list.txt").read().splitlines()

    def load_training_dataset(self):
        return open("toy_data/verb_only/constrained_training_data.txt").readlines()

    def preprocess(self, text):
        lemma = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]

        tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
        return " ".join(tokens)

    def get_proportional_reward(self, action_index):
        if self.action_idx2freq[action_index] < self.action_threshold:
            return 1
        else:
            return 1 / self.imbalance_ratio
        # print(action_index)
        # return 1 / self.action_idx2freq[action_index]
        # if self.action_idx2freq[action_index] < self.action_threshold:
        #     return torch.FloatTensor(1)

        # return torch.FloatTensor(1 / self.action_idx2freq[action_index])

    def _get_next_q_value(self, state_rep, q_values):
        chosen_index = torch.argmax(self.model.softmax(q_values), dim=1)
        # action_verb_index = self.word2index[
        #     self.action2index[chosen_index.item()]]
        action_verb_index = self.word2index[
            self.all_actions[chosen_index.item()]]
        comb_state_rep = state_rep.copy()
        comb_state_rep.append(action_verb_index)
        with torch.no_grad():
            statept = torch.LongTensor(comb_state_rep)
            state_embeds = self.model.get_embeds(statept)
            x = self.model.get_encoding(state_embeds.unsqueeze(0))
            q_values = self.model.predicate_pointer(x)
        max_q = q_values[0][q_values.argmax(1)]
        return max_q

    def model_make_step(self, state_rep, data_index, correct_indices, epsilon, final_step=False):
        state_embeds = self.model.get_embeds(state_rep)
        state_embeds = torch.FloatTensor(state_embeds)
        x = self.model.get_encoding(state_embeds.unsqueeze(0))
        q_values = self.model.predicate_pointer(x)
        next_max_q = self._get_next_q_value(state_rep, q_values)

        if random.random() < epsilon:
            chosen_index = random.randrange(0, len(self.all_actions) - 1)
        else:
            chosen_index = torch.argmax(
                self.model.softmax(q_values), dim=1).item()
        max_q = q_values.squeeze()[chosen_index]

        if self.action_idx2freq[chosen_index] > self.action_threshold:
            reward = -1 / self.imbalance_ratio
        else:
            reward = -1
        for i, fact_idxs in enumerate(correct_indices):
            if chosen_index in fact_idxs:
                reward = self.get_proportional_reward(chosen_index)
                correct_indices.pop(i)
                break

        # for Logging purposes
        if reward > 0:
            pure_reward = 1
        else:
            pure_reward = -1

        reward = reward + self.gamma * next_max_q * (1 - final_step)

        loss = self.model.loss_f(max_q, reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return chosen_index, (loss, pure_reward), correct_indices

    def train_QA(self):
        total_frames = 0
        # accuracy_collection = []
        # loss_collection = []
        self.model.train()
        with open('toy_data/verb_only/accuracy.txt', 'w') as acc_file, open("toy_data/verb_only/loss.txt", "w") as loss_file:
            print("emptied previous file")
        episodes_so_far = 1
        for epoch in range(1, self.num_epochs + 1):
            epoch_accuracies = list()
            epoch_losses = list()
            for data_index in range(len(self.training_dataset)):
                frame_accuracies = list()
                frame_losses = list()
                for episode_idx in range(self.num_episodes):
                    row = json.loads(self.training_dataset[data_index])
                    question = row['question']
                    choices = row['choices']
                    correct_indices = self.get_action_indices(row)
                    epsilon = self.epsilon_scheduler.value(episodes_so_far)

                    pred_indices = list()
                    pred_strings = list()
                    state_reps = list()

                    """
                        Step 1
                    """
                    state_rep1 = row['formatted_question']
                    state_reps.append(state_rep1)

                    chosen_index1, (loss1, accuracy1), correct_indices = self.model_make_step(
                        state_rep1, data_index, correct_indices, epsilon, final_step=False)

                    pred_indices.append(chosen_index1)
                    pred_strings.append(self.all_actions[pred_indices[-1]])

                    # self.add_loss(loss1)
                    # self.add_accuracy(accuracy1)
                    frame_accuracies.append(accuracy1)
                    frame_losses.append(loss1)
                    """
                        Step 2
                    """
                    state_rep2 = state_rep1 + f" {pred_strings[-1]}"
                    state_reps.append(state_rep2)

                    chosen_index2, (loss2, accuracy2), correct_indices = self.model_make_step(
                        state_rep2, data_index, correct_indices, epsilon, final_step=True)

                    pred_indices.append(chosen_index2)
                    pred_strings.append(
                        self.all_actions[int(pred_indices[-1])])
                    # self.add_loss(loss2)
                    # self.add_accuracy(accuracy2)
                    frame_accuracies.append(accuracy2)
                    frame_losses.append(loss2)

                self.add_loss(sum(frame_losses) / len(frame_losses))
                self.add_accuracy(sum(frame_accuracies) /
                                  len(frame_accuracies))
                avg_loss, avg_acc = self.get_avg_loss_acc()
                epoch_accuracies.append(avg_acc)
                epoch_losses.append(avg_loss)
                episodes_so_far += 1
                print(
                    f"{epoch}|{question},  chosen preds:{pred_strings[-2]}, {pred_strings[-1]}, epsilon {epsilon}, avg loss: {avg_loss}, avg_acc: {avg_acc}")
            with open('toy_data/verb_only/accuracy.txt', 'a+') as acc_file, open("toy_data/verb_only/loss.txt", "a+") as loss_file:
                acc_file.write(
                    f"{round(float(sum(epoch_accuracies) / len(epoch_accuracies)), 2)}\n")
                loss_file.write(
                    f"{round(float(sum(epoch_losses) / len(epoch_losses)), 2)}\n")
            # accuracy_collection.append(
            #     sum(epoch_accuracies) / len(epoch_accuracies))
            # loss_collection.append(sum(epoch_losses) / len(epoch_losses))

        torch.save(self.model.state_dict(), f'constrained_verb_only.pt')
        # accuracies = [round(float(acc), 2) for acc in accuracies]
        # losses = [round(float(lss), 2) for lss in losses]
        # plt.plot(accuracies)
        # plt.plot(losses)
        # plt.legend(['accuracy', 'loss'], loc='upper left')
        # plt.xlabel('epochs')
        # plt.title(self.experiment_name)
        # plt.savefig("plots/" + self.experiment_name + ".png")
        # plt.show()

    def get_answer_letter(self, answer_index):
        switcher = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H"
        }
        return switcher.get(answer_index)

    def get_correct_one_hot_encoded_action_indices(self, row):
        correct_actions = set()
        if 'fact1_pred' not in row and 'fact2_pred' not in row:
            return None
        if 'fact1_pred' in row:
            for pred in row['fact1_pred']:
                correct_actions.add(pred)
        if 'fact2_pred' in row:
            for pred in row['fact2_pred']:
                correct_actions.add(pred)

        correct_action_indices = list()
        zeros = torch.zeros(1, len(self.all_actions), dtype=torch.float)
        for action in correct_actions:
            zeros[0][self.action2index[action]] = 1
        return zeros

    def get_action_indices(self, row):
        """
        @return: a 2d array, first index specifies fact, second specifies the associated predicates
        """
        if len(row['fact1_pred']) == 0 and len(row['fact2_pred']) == 0:
            return 0
        combined_actions = list()
        actions1 = list()
        for pred in row['fact1_pred']:
            actions1.append(self.action2index[pred])
        actions2 = list()
        for pred in row['fact2_pred']:
            actions2.append(self.action2index[pred])
        combined_actions.append(actions1)
        combined_actions.append(actions2)
        return combined_actions

    def load_action2freq(self):
        pred_counter = defaultdict(int)
        for i in range(len(self.training_dataset)):
            row = json.loads(self.training_dataset[i])
            if 'fact1_pred' in row:
                for pred in row['fact1_pred']:
                    pred_counter[pred] += 1
            if 'fact2_pred' in row:
                for pred in row['fact2_pred']:
                    pred_counter[pred] += 1

        idx2freq = dict()
        for pred in pred_counter.keys():
            key = self.action2index[pred]
            idx2freq[key] = pred_counter[pred]

        return idx2freq
