from basic_dqn import DQNTrainer
from joblib import Parallel, delayed
import multiprocessing
import gc


def parallelize(game, params):
    print(params)
    trainer = DQNTrainer(game, params)
    trainer.train()


if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.train_QA()
