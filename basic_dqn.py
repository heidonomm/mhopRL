import math
import random
import textworld
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchnlp.nn import Attention


from collections import deque
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

# from matplotlib import use
# use('Agg')
import matplotlib.pyplot as plt

import logging

from replay import *
from schedule import *
from utils import NegativeLogLoss, words_to_ids, to_one_hot

# from memory_profiler import profile

USE_CUDA = torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        print(f"is cuda available: {USE_CUDA}")

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.embeddings = nn.Embedding(self.num_inputs, 16)
        self.encoder = nn.LSTM(16, 8, 2, batch_first=True)
        self.relu = nn.ReLU()

        self.aggregator = nn.LSTM(8, 8, batch_first=True)
        self.predicate_pointer = nn.Linear(8, self.num_actions)
        self.answer_pointer = nn.Linear(8, 8)
        self.softmax = nn.Softmax(dim=1)

        self.loss_f = nn.BCEWithLogitsLoss()

    def get_encoding(self, x):
        # embs = torch.reshape(
        #     embs, (embs.size(1), embs.size(0), embs.size(2)))
        encoding, _ = self.encoder(x)
        x = self.relu(encoding)
        x, _ = self.aggregator(x)
        x = self.relu(x)
        return x[:, -1, :]

    def get_embeds(self, inputLong):
        return self.embeddings(inputLong)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            statept = torch.LongTensor(state)
            state_embeds = self.get_embeds(statept).unsqueeze(0)
            with torch.no_grad():
                x = self.get_encoding(state_embeds)
                q_values = self.predicate_pointer(x)
                action = self.softmax(q_values)
                action_index = torch.argmax(action, dim=1)
        else:
            action_index = random.randrange(self.num_actions)
        return action_index

    def answer_question(self, state):
        statept = torch.LongTensor(state)
        embs = self.get_embeds(statept).unsqueeze(0)
        with torch.no_grad():
            x = self.get_encoding(embs)
            answer_values = self.answer_pointer(x)
            norm_values = self.softmax(answer_values)
        return torch.argmax(norm_values, dim=1)


class DQNTrainer(object):
    def __init__(self):
        self.num_epochs = 32

        self.vocab = self.load_vocab()
        self.word2index = self.load_word2index()
        self.all_actions = self.load_action_dictionary()
        self.training_dataset = self.load_training_dataset()
        self.pred2index = self.load_action2index()

        self.model = DQN(len(self.vocab), len(
            self.all_actions))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

        self.erroneous_facts = set()

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
        return open("word_vocab.txt").read().splitlines()

    def load_action_dictionary(self):
        return open("all_predicates.txt").read().splitlines()

    def load_training_dataset(self):
        return open("toy_data/constrained_training_data.txt").readlines()

    def plot(self, frame_idx, rewards, losses, completion_steps):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. steps: %s' %
                  (frame_idx, np.mean(completion_steps[-10:])))
        plt.plot(completion_steps)
        plt.subplot(133)
        plt.title('loss-dqn')
        plt.plot(losses)
        # txt = "Gamma:" + str(self.gamma) + ", Num Frames:" + str(self.num_frames) + ", E Decay:" + str(epsilon_decay)
        plt.figtext(0.5, 0.01, self.filename, wrap=True,
                    horizontalalignment='center', fontsize=12)
        # plt.show()
        fig.savefig('plots/' + self.filename + '_' + str(frame_idx) + '.png')

    def preprocess(self, text):
        lemma = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]

        tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
        return " ".join(tokens)

    def model_make_step(self, state_rep, data_index, correct_indices):
        statept = torch.LongTensor(state_rep)
        state_embeds = self.model.get_embeds(statept)
        x = self.model.get_encoding(state_embeds.unsqueeze(0))
        q_values = self.model.predicate_pointer(x)

        one_hot_y_true = self.get_correct_one_hot_encoded_action_indices(
            json.loads(self.training_dataset[data_index]))

        if one_hot_y_true is None:
            return 0
        if len(one_hot_y_true) == 0:
            return 0

        loss = self.model.loss_f(q_values, one_hot_y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        chosen_index = torch.argmax(self.model.softmax(q_values), dim=1)

        reward = 0
        for i, fact_idxs in enumerate(correct_indices):
            if str(chosen_index.item()) in fact_idxs:
                reward = 1
                correct_indices.pop(i)

        return chosen_index, (loss, reward), correct_indices

    def train_QA(self):
        total_frames = 0
        accuracy_collections = []
        loss_arr = []
        for epoch in range(1, self.num_epochs + 1):
            for data_index in range(len(self.training_dataset)):
                self.model.zero_grad()
                row = json.loads(self.training_dataset[data_index])
                question = row['question']
                choices = row['choices']
                correct_indices = self.get_action_indices(row)

                pred_indices = list()
                pred_strings = list()
                state_reps = list()

                """
                    Step 1
                """
                state_rep1 = words_to_ids(self.preprocess(
                    row['formatted_question']), self.word2index)
                state_reps.append(state_rep1)

                chosen_index1, (loss1, accuracy1), correct_indices = self.model_make_step(
                    state_rep1, data_index, correct_indices)

                pred_indices.append(chosen_index1)
                pred_strings.append(self.all_actions[pred_indices[-1]])
                self.add_loss(loss1)
                self.add_accuracy(accuracy1)

                """
                    Step 2
                """
                state_rep2 = state_rep1 + \
                    words_to_ids(f" {pred_strings[-1]}", self.word2index)
                state_reps.append(state_rep2)

                chosen_index2, (loss2, accuracy2), correct_indices = self.model_make_step(
                    state_rep2, data_index, correct_indices)

                pred_indices.append(chosen_index2)
                pred_strings.append(self.all_actions[int(pred_indices[-1])])
                self.add_loss(loss2)
                self.add_accuracy(accuracy2)

                avg_loss, avg_acc = self.get_avg_loss_acc()

                if avg_acc > 0 or accuracy2 == 1 or accuracy1 == 1:
                    with open("it_works.txt", "w") as works:
                        works.write("it did something")
                print(
                    f"{epoch}|{question}  ,  chosen preds:{pred_strings[-2]}, {pred_strings[-1]}, avg loss: {avg_loss}, avg_acc: {avg_acc}")

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
            zeros[0][self.pred2index[action]] = 1
        return zeros

    def get_action_indices(self, row):
        """
        @return: a 2d array, first index specifies fact, second specifies the associated predicates
        """
        if len(row['fact1_pred']) == 0 and len(row['fact2_pred']) == 0:
            self.erroneous_facts.add(json.dumps(row))
            return 0
        actions = []
        actions += row['fact1_pred']
        actions += row['fact2_pred']
        return actions
