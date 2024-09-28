from collections import defaultdict
import numpy as np


class RandomAgent:
    """
    We need to collect sample data from agent's behavior.
    RandomAgent follows random policy pi(a|s) = {up: 0.25, down: 0.25, left: 0.25, right: 0.25}

    """

    def __init__(self, n_action, gamma=0.9):
        """
        n_action : number of available actions that agent can do
        gamma : discount rate

        self.pi: pi(a}s)
        self.V : V(s) under random policy
        self.episode : record (s,a,r) for an episode

        """
        self.gamma = gamma
        self.n_action = n_action
        p = 1 / n_action

        # actions disctribution
        random_actions = {i: p for i in range(n_action)}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.episode = []  # records (s, a, r)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def record(self, state, action, reward):
        data = (state, action, reward)
        self.episode.append(data)

    def reset(self):
        self.episode.clear()  # clear memory list

    def evaluate(self):
        """
        compute G(return) efficiently

        episode: state A -> action -> state B -> action -> state C -> action -> goal
                                       (R0)                 (R1)                 (R2)
        G_A = R0 + gamma*R1 + gamma^2*R2
        G_B = R1 + gamma*R2
        G_C = R2

        when calculate G_C -> G_B -> G_A reversly, we can remove the duplicated commputation.

        V(s) <- (G0 + G1 + ... Gn) / n
             <- V(s) + (Gn - V(s)) / n
        """
        G = 0
        # reversely compute G
        for data in reversed(self.episode):
            state, action, reward = data
            G = reward + self.gamma * G
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]
