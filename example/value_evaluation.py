if "__file__" in globals():
    import os, sys

    # os.path.dirname(__file__) : return 현재 실행중인 파이썬 파일의 path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from common.gridworld import GridWorld
from models.DP import value_iteration, greedy_policy
from collections import defaultdict


V = defaultdict(lambda: 0)
env = GridWorld()
gamma = 0.9

# value iteration : get optimal V(s)
V = value_iteration(V, env, gamma, is_render=True)

optimal_pi = greedy_policy(V, env, gamma)
# optimal policy visualization
env.render_v(V, optimal_pi)
