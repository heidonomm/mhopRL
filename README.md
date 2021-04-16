# Reinforcement Learning agent for multihop reasoning on QASC dataset

Implementation of a Deep Q-Learning for a multihop reasoning dataset QASC, where an agent is tasked with choosing two facts from a large corpus to pick a correct answer for a specified question.

For example, for the question "What can be used for transportation?" with the possible answers of "(A) Trailers and boats, (B) hitches, (C) trees and flowers, (D) air masses, (E) Cats", the agent may retrieve facts  “Trailers and boats are counted as a private vehicles”, and “A vehicle is used for transportation.”, which directs the agent to choose "A" as its final answer.

---

Dependecies:
`yaml, sentence-transformers, torch, nltk, numpy, matplotlib`

---

Model can be run with `train.py` and evaluation of the trained model (on the same training files) can be performed with `analyse.py`, which generates analysis files outlining which questions get answered correctly, which ones falsely and for which ones is the model able to choose two different verbs for.
