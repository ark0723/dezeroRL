from collections import defaultdict
from common.utils import argmax


### policy evaluation ###
def evaluate_one_step(pi, V, env, gamma=0.9):
    for state in env.states():  # 각 상태에 접근
        # 목표 상태에서의 가치함수는 항상 0 : 목표상태에서는 에피소드가 끝나고 그 다음 전개는 없기 때문
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        expected_value = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            reward = env.get_reward(next_state)
            expected_value += action_prob * (reward + gamma * V[next_state])

        V[state] = expected_value

    return V


def policy_evaluate(pi, V, env, gamma, epsilon=1e-3):
    while True:
        old_V = V.copy()
        V = evaluate_one_step(pi, V, env, gamma)

        # 갱신된 양의 최댓값 계산
        delta = 0
        for state in V.keys():
            delta = max(delta, abs(V[state] - old_V[state]))

        if delta < epsilon:
            break
    return V


### 평가와 개선의 반복: find optimal policy (policy iteration)
def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        Q = {}  # action-value
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.get_reward(next_state)
            Q[action] = reward + gamma * V[next_state]
        max_action = argmax(Q)
        # set the next pi(a|s) as the max_action
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iteration(env, gamma, epsilon=1e-3, is_render=False):
    # inital policy: evenly random actions 각 행동이 균등하게 선택되도록
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        # 1. evaluation : 현재 정책 평가
        V = policy_evaluate(pi, V, env, gamma, epsilon)
        # 2. greedy algorithm : 정책 개선 (현재 정책 -> 새로운 정책)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        # 3. check if new policy is updated or not
        if new_pi == pi:  # optimal policy
            break
        pi = new_pi

    return pi


### value iteration ###
def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0  # 목표 상태에서의 가치 함수는 항상 0
            continue
        Q = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.get_reward(next_state)
            Q.append(reward + gamma * V[next_state])
        V[state] = max(Q)
    return V


def value_iteration(V, env, gamma, epsilon=1e-3, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()  # 갱신 전 가치함수
        V = value_iter_onestep(V, env, gamma)  # 갱신

        # 갱신된 양의 최댓값 계산
        delta = 0
        for state in V.keys():
            delta = max(delta, abs(V[state] - old_V[state]))

        # check stop point
        if delta < epsilon:
            break
    return V
