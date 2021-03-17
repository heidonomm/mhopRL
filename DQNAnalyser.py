import json
from collections import defaultdict

import nltk
import torch

from utils import words_to_ids
from dqn import DQNTrainer


class DQNAnalyser(DQNTrainer):
    def __init__(self):
        super().__init__()

        self.model.load_state_dict(torch.load('constrained_verb_only.pt'))

        self.action_predicted_correctly = defaultdict(int)
        self.action_predicted_falsely = defaultdict(int)

        self.questions_predicted_correctly = dict()
        self.questions_predicted_falsely = dict()
        self.questions_where_predictions_are_different = dict()

    def load_action_dictionary(self):
        return open("toy_data/verb_only/constrained_predicate_list.txt").read().splitlines()

    def load_training_dataset(self):
        return open("toy_data/verb_only/constrained_training_data.txt").readlines()

    def model_make_step(self, state_rep, data_index, correct_indices):
        with torch.no_grad():
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

            chosen_index = torch.argmax(self.model.softmax(q_values), dim=1)

            reward = 0
            for i, fact_idxs in enumerate(correct_indices):
                if chosen_index.item() in fact_idxs:
                    reward = 1
                    correct_indices.pop(i)

            return chosen_index, reward, correct_indices

    def evaluate(self):
        self.model.eval()
        for data_index in range(len(self.training_dataset)):
            row = json.loads(self.training_dataset[data_index])
            question = row['question']
            print(question)
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

            chosen_index1, reward1, correct_indices = self.model_make_step(
                state_rep1, data_index, correct_indices)

            pred_indices.append(chosen_index1)
            pred_strings.append(self.all_actions[chosen_index1])
            if reward1 == 1:
                self.action_predicted_correctly[pred_strings[-1]] += 1
            else:
                self.action_predicted_falsely[pred_strings[-1]] += 1

            """
                Step 2
            """
            state_rep2 = state_rep1 + \
                words_to_ids(f" {pred_strings[-1]}", self.word2index)
            state_reps.append(state_rep2)

            chosen_index2, reward2, correct_indices = self.model_make_step(
                state_rep2, data_index, correct_indices)

            pred_indices.append(chosen_index2)
            pred_strings.append(
                self.all_actions[int(pred_indices[-1])])

            if reward2 == 1:
                self.action_predicted_correctly[pred_strings[-1]] += 1
            else:
                self.action_predicted_falsely[pred_strings[-1]] += 1

            if reward1 == 1 and reward2 == 1:
                self.questions_predicted_correctly[question] = (
                    pred_strings[-2], pred_strings[-1])
            if reward1 == 0 and reward2 == 0:
                self.questions_predicted_falsely[question] = (
                    pred_strings[-2], pred_strings[-1])
            if pred_strings[-2] != pred_strings[-1]:
                self.questions_where_predictions_are_different[question] = (
                    pred_strings[-2], pred_strings[-1])

        print(len(self.training_dataset))
        self.write_analysis_files()

    def write_analysis_files(self):
        with open("toy_data/verb_only/analysis_files/predicate_counts.txt", "w") as out_file:
            correct_keys = self.action_predicted_correctly.keys()
            for key in correct_keys:
                if key in self.action_predicted_falsely.keys():
                    out_file.write(
                        f"{key} | wrongly: {self.action_predicted_falsely[key]}, correctly: {self.action_predicted_correctly[key]}\n")
                else:
                    out_file.write(
                        f"{key} | correctly: {self.action_predicted_correctly[key]}\n")
            for key in self.action_predicted_falsely.keys():
                if key not in correct_keys:
                    out_file.write(
                        f"{key} | correctly: {self.action_predicted_falsely[key]}\n")

        with open("toy_data/verb_only/analysis_files/q_all_correct.txt", "w") as out_file:
            for question in self.questions_predicted_correctly.keys():
                out_file.write(
                    f"{question} | {self.questions_predicted_correctly[question]}\n")

        with open("toy_data/verb_only/analysis_files/q_all_wrong.txt", "w") as out_file:
            for question in self.questions_predicted_falsely.keys():
                out_file.write(
                    f"{question} | {self.questions_predicted_falsely[question]}\n")

        with open("toy_data/verb_only/analysis_files/q_different_predictions.txt", "w") as out_file:
            for question in self.questions_where_predictions_are_different.keys():
                out_file.write(
                    f"{question} | {self.questions_where_predictions_are_different[question]}\n")
