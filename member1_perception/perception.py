"""Member 1: CNN (frame -> entity/player detections) + LSTM (detections over
time -> tracked entities & predicted trajectories for moving platforms/enemies).
Implement PerceptionModule from core.interfaces. Do not import from the other
3 member packages.
"""
from core.interfaces import PerceptionModule, PerceptionOutput, Frame


class CNNLSTMPerception(PerceptionModule):
    def __init__(self, history_len: int = 8, horizon: int = 10):
        self.history_len = history_len
        self.horizon = horizon
        # TODO: load/define CNN backbone (e.g. small ResNet or custom conv stack)
        # TODO: define LSTM over CNN feature sequence for trajectory prediction

    def process(self, frame: Frame, history: list[Frame]) -> PerceptionOutput:
        raise NotImplementedError
