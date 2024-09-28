if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.gridworld import GridWorld
from models.montecarlo import RandomAgent
import numpy as np

env = GridWorld()
agent = RandomAgent(n_action=4)

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    agent.reset()  # reset episode record

    while True:
        action = agent.get_action(state)  # Agent: select an action
        next_state, reward, done = env.step(action)  # Environment: state transition
        agent.record(state, action, reward)  # save tuple (state, action, reward)
        # reach the goal
        if done:
            # calculate V(s) for all states that included in an episode
            agent.evaluate()
            break

        state = next_state

# visualize V(s)
env.render_v(agent.V)
