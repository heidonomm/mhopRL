import math

"""
Original schedule.py taken from https://github.com/rajammanabrolu/KG-DQN
Adapted from https://github.com/berkeleydeeprlcourse/homework
"""

class ExponentialSchedule(object):
    def __init__(self, schedule_timesteps, decay, final_e, initial_e=1.0):
        self.decay = decay
        self.initial_e = initial_e
        self.final_e = final_e
        self.schedule_timesteps = schedule_timesteps

    def value(self, t):
        return self.final_e + (self.initial_e - self.final_e) * math.exp(-1. * t / self.decay)
