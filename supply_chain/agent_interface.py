import abc

class AgentInterface(abc.ABC):

    @abc.abstractclassmethod
    def choose_action(self, state: object):
        pass

    @abc.abstractclassmethod
    def update(self, state: object, action: object, next_state: object, reward: float):
        pass

    @abc.abstractclassmethod
    def train(self, num_episodes: int, greedy: bool = False):
        pass