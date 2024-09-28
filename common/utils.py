def argmax(action_dict):
    """
    example:
        input: action_dict = {0:0.1, 1: -0.3, 2:9.9, 3: -1.3}
        return: 2
    """
    max_action = max(action_dict, key=action_dict.get)
    return max_action
