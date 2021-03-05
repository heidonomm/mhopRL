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
    # @profile()
    def __init__(self):
        self.num_epochs = 32
        self.update_freq = 4
        self.filename = 'dqn_' + '_'.join(['Testing'])
        logging.basicConfig(filename='logs/' + self.filename +
                            '.log', level=logging.WARN, filemode='w')

        self.replay_buffer = ReplayBuffer(100)

        self.vocab = self.load_vocab()
        self.word2index = self.load_word2index()
        self.all_actions = self.load_action_dictionary()
        self.training_dataset = self.load_training_dataset()
        self.pred2index = self.load_action2index()

        self.model = DQN(len(self.vocab), len(
            self.all_actions))
        # model = nn.DataParallel(model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

        self.num_frames = 100
        self.batch_size = 1
        self.gamma = 0.2

        self.losses = []
        self.all_rewards = []
        self.completion_steps = []
        self.lowest_loss = 100

        self.rho = 0.25

        self.nr_of_frames = len(self.training_dataset) * self.num_epochs
        self.e_scheduler = LinearSchedule(self.nr_of_frames, 0.01)

        self.queue_size = 50
        self.loss_queue = []
        self.accuracy_queue = []

    def add_loss(self, loss_value):
        if len(self.loss_queue) > self.queue_size:
            self.loss_queue.pop(0)
            self.loss_queue.append(loss_value)
        else:
            self.loss_queue.append(loss_value)

    def add_accuraccy(self, acc_value):
        if len(self.accuracy_queue) > self.queue_size:
            self.accuracy_queue.pop(0)
            self.accuracy_queue.append(acc_value)
        else:
            self.accuracy_queue.append(acc_value)

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
        return open("toy_data/training_data.txt").readlines()

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

    def compute_td_loss(self):
        # var = self.replay_buffer.sample(self.batch_size, self.rho)
        state, action, reward, next_state, done, data_index = self.replay_buffer.sample(
            self.batch_size, self.rho)
        statept = torch.LongTensor(state)
        state_embeds = self.model.get_embeds(statept)
        next_statept = torch.LongTensor(next_state)
        next_state_embeds = self.model.get_embeds(next_statept)
        # actionpt = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)

        # statept.requires_grad = True
        # next_statept.requires_grad = True
        done = torch.FloatTensor(1 * done)
        x = self.model.get_encoding(state_embeds)
        q_values = self.model.predicate_pointer(x)
        # action_distribution = self.model.softmax(q_values)
        # q_value = torch.argmax(action, dim=1)

        # with torch.no_grad():
        #     next_x = self.model.get_encoding(state_embeds)
        #     next_q_values = self.model.predicate_pointer(x)
        #     next_action = self.model.softmax(q_values)
        #     next_q_value = torch.argmax(next_action, dim=1)
        # q_value = self.model.act(statept, 0)
        # next_q_value = self.model.act(next_state, 0)
        correct_indices = self.get_correct_one_hot_encoded_action_indices(
            json.loads(self.training_dataset[data_index[0]]))
        if correct_indices is None:
            return 0
        if len(correct_indices) == 0:
            return 0
        loss = self.model.loss_f(
            q_values, correct_indices)
        # loss = NegativeLogLoss(action_distribution, correct_indices)
        # loss = torch.mean(loss)

        # expected_q_value = reward + \
        #     (self.gamma * next_q_value).type(torch.FloatTensor) * (1 - done)
        # loss = (q_value - (expected_q_value.data)).pow(2).mean()
        # # clipped_loss = loss.clamp(-1.0, 1.0)
        # loss = loss.clamp(-1.0, 1.0)
        # right_gradient = clipped_loss * -1.0
        # print(loss)

        self.optimizer.zero_grad()
        # loss.backward(right_gradient.data.unsqueeze(1)[:, 0])
        loss.backward()

        self.optimizer.step()

        return loss

    def preprocess(self, text):
        lemma = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]

        tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
        return " ".join(tokens)

    def state_rep_generator(self, q_obj):
        state = ""
        elements = list()
        elements.append(preprocess(q_obj["stem"]))
        for choice in q_obj['choices']:
            elements.append((preprocess(choice['text'])))
        return "|".join(elements)

    # def predict_get_loss_accuraccy(self, )

    def add_state_rep(self, state_rep, fact):
        norm_fact = self.preprocess(fact)
        state_rep += f"<|>{norm_fact}"
        return state_rep

    def calculate_reward(self, row, prediction):
        y_true = list()
        if 'fact1_pred' in row:
            y_true += row['fact1_pred']
        if 'fact2_pred' in row:
            y_true += row['fact2_pred']
        if len(y_true) == 0:
            with open("empty_facts.jsonl", 'a+') as empty_facts:
                empty_facts.write(f"{json.dumps(row)}\n")
            return 0
        if prediction in y_true:
            return 1
        else:
            return -1

    def model_make_step(self, state_rep, epsilon, data_index):
        statept = torch.LongTensor(state_rep)
        state_embeds = self.model.get_embeds(statept)
        x = self.model.get_encoding(state_embeds)
        q_values = self.model.predicate_pointer(x)

        one_hot_y_true = self.get_correct_one_hot_encoded_action_indices(
            json.loads(self.training_dataset[data_index]))

        if one_hot_y_true is None:
            return 0
        if len(one_hot_y_true) == 0:
            return 0

        loss = self.model.loss_f(q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        chosen_index = torch.argmax(self.softmax(q_values), dim=1)

        if chosen_index in correct_indices:
            reward = 1
        else:
            reward = 0

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
                # print(question)

                # state_rep = self.state_rep_generator(row['formatted_question'])
                episode_reward = 0

                pred_indices = list()
                pred_strings = list()
                rewards = list()
                state_reps = list()

                state_rep1 = words_to_ids(self.preprocess(
                    row['formatted_question']), self.word2index)
                state_reps.append(state_rep1)
                # epsilon = self.e_scheduler.value(data_index)
                epsilon = 0
                pred_indices.append(self.model.act(
                    state_reps[-1], epsilon))
                pred_strings.append(self.all_actions[pred_indices[-1]])
                rewards.append(self.calculate_reward(row, pred_strings[-1]))
                # state_rep2 = torch.cat((state_rep1,
                #                         words_to_ids(f" {pred_strings[-1]}", self.word2index)))
                state_rep2 = state_rep1 + \
                    words_to_ids(f" {pred_strings[-1]}", self.word2index)
                state_reps.append(state_rep2)

                print(f"question:{question}, predicate:{pred_strings[-1]}\n")
                # exper = Experience(
                #     state=state_reps[-2], action=pred_indices[-1], reward=rewards[-1], next_state=state_reps[-1], done=False)
                # self.replay_buffer.push(exper)
                self.replay_buffer.push(
                    state_reps[-2], pred_indices[-1], rewards[-1], state_reps[-1], False, data_index)

                logging.info('-------')
                logging.info(question)
                logging.info(pred_strings[-1])

                pred_indices.append(self.model.act(
                    state_reps[-1], epsilon))
                pred_strings.append(self.all_actions[int(pred_indices[-1])])
                # state_rep3 = torch.cat((state_rep2,
                #                         words_to_ids(f" {pred_strings[-1]}", self.word2index)))
                state_rep3 = state_rep2 + \
                    words_to_ids(f" {pred_strings[-1]}", self.word2index)
                state_reps.append(state_rep3)
                rewards.append(self.calculate_reward(row, pred_strings[-1]))

                logging.info('-------')
                logging.info(question)
                logging.info(pred_strings[-1])

                answer_index = self.model.answer_question(
                    state_reps[-1]).item()
                pred_indices.append(answer_index)
                pred_strings.append(self.get_answer_letter(pred_indices[-1]))
                # state_rep4 = torch.cat((state_rep3, words_to_ids(
                #     f" {pred_strings[-1]}", self.word2index)))
                state_rep4 = state_rep3 + words_to_ids(
                    f" {pred_strings[-1]}", self.word2index)
                state_reps.append(state_rep4)
                # exper = Experience(
                #     state=state_reps[-2], action=pred_indices[-1], reward=rewards[-1], next_state=state_reps[-1], done=True)
                # self.replay_buffer.push(exper)
                self.replay_buffer.push(
                    state_reps[-2], pred_indices[-1], rewards[-1], state_reps[-1], True, data_index)

                if len(self.replay_buffer) > self.batch_size:
                    loss = self.compute_td_loss()
                    if loss < self.lowest_loss:
                        self.lowest_loss = loss
                        self.losses.append(loss.item())
                        print(f"current lowest loss is: {self.lowest_loss}")
                    if loss != 0:
                        self.losses.append(loss.item())
            # @profile()
        print(f"loss at the end of training is: {self.lowest}")
        print(self.losses)

    def train(self):
        total_frames = 0
        for e_idx in range(1, self.num_epochs + 1):
            state = self.env.reset()
            state_text = state.description
            state_rep = self.state_rep_generator(state_text)
            episode_reward = 0
            completion_steps = 0
            episode_done = False

            for frame_idx in range(1, self.num_frames + 1):
                epsilon = self.e_scheduler.value(total_frames)
                action = self.model.act(state_rep, epsilon)

                action_text = self.all_actions[int(action)]
                logging.info('-------')
                logging.info(state_text)
                logging.info(action_text)

                next_state, reward, done = self.env.step(action_text)
                reward += next_state.intermediate_reward
                reward = max(-1.0, min(reward, 1.0))

                # if reward != 0:
                logging.warning('--------')
                logging.warning(frame_idx)
                logging.warning(state_text)
                # print(next_state_text)
                logging.warning(action_text)
                logging.warning(reward)

                # print(reward)

                next_state_text = next_state.description
                next_state_rep = self.state_rep_generator(next_state_text)

                self.replay_buffer.push(
                    state_rep, action, reward, next_state_rep, done)

                state = next_state
                state_text = next_state_text
                state_rep = next_state_rep

                episode_reward += reward
                completion_steps += 1
                total_frames += 1

                if len(self.replay_buffer) > self.batch_size:
                    if frame_idx % self.update_freq == 0:
                        loss = self.compute_td_loss()
                        self.losses.append(loss.data[0])

                if done:
                    logging.warning("Done")
                    state = self.env.reset()
                    state_text = state.description
                    state_rep = self.state_rep_generator(state_text)
                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0
                    episode_done = True
                elif frame_idx == self.num_frames:

                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0

                if episode_done:
                    break

            if e_idx % (int(self.num_episodes / 10)) == 0:
                logging.info("Episode:" + str(e_idx))
                self.plot(e_idx, self.all_rewards,
                          self.losses, self.completion_steps)
                self.plot(e_idx, self.all_rewards,
                          self.losses, self.completion_steps)
                parameters = {
                    'model': self.model,
                    'replay_buffer': self.replay_buffer,
                    'action_dict': self.all_actions,
                    'vocab': self.vocab,
                    'params': self.params,
                    'stats': {
                        'losses': self.losses,
                        'rewards': self.all_rewards,
                        'completion_steps': self.completion_steps
                    }
                }
                torch.save(parameters, 'models/' +
                           self.filename + '_' + str(e_idx) + '.pt')

        parameters = {
            'model': self.model,
            'replay_buffer': self.replay_buffer,
            'action_dict': self.all_actions,
            'vocab': self.vocab,
            'params': self.params,
            'stats': {
                'losses': self.losses,
                'rewards': self.all_rewards,
                'completion_steps': self.completion_steps
            }
        }
        torch.save(parameters, 'models/' + self.filename + '_final.pt')
        self.env.close()

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
            return 0
        actions = []
        actions.append[row['fact1_pred']]
        actions.append[row['fact2_pred']]
        return actions
