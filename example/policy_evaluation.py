if "__file__" in globals():
    import os, sys

    # os.path.dirname(__file__) : return 현재 실행중인 파이썬 파일의 path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.gridworld import GridWorld
from models.DP import policy_iteration

env = GridWorld()
gamma = 0.9
pi = policy_iteration(env, gamma, is_render=True)
