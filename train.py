from DQNTrainer import DQNTrainer
from joblib import Parallel, delayed
import multiprocessing
import gc

if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.train()
