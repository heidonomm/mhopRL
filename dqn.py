import random

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

USE_CUDA = torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(DQN, self).__init__()
        print(f"is cuda available: {USE_CUDA}")

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.embeddings = SentenceTransformer('nq-distilbert-base-v1')
        # self.embeddings.weight.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = False
        self.embedding_size = 768
        self.net = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_actions)
        )
        # self.encoder = nn.LSTM(768, hidden_size, 2, batch_first=True)

        # self.predicate_pointer = nn.Linear(hidden_size, self.num_actions)
        # self.answer_pointer = nn.Linear(hidden_size, 8)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.loss_f = nn.SmoothL1Loss()

    def get_encoding(self, x):
        # embs = torch.reshape(
        #     embs, (embs.size(1), embs.size(0), embs.size(2)))
        encoding, _ = self.encoder(x)
        x = self.relu(encoding)
        x, _ = self.aggregator(x)
        x = self.relu(x)
        return x[:, -1, :]

    def get_embeds(self, inputString):
        return self.embeddings.encode(inputString)

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
