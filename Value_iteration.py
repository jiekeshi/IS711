from re import A
from time import time


### Global Settings

convergence_tol = 0.01
discount_factor = 0.95

### MDP classes

class State:
    def __init__(self, pos):
        self.pos = pos
        self.value = 0
        self.value_old = 0
        self.is_converged = True
        self.policy = "--"

class Transition:
    def __init__(self, start, end, transition_prob, reward):
        self.start = start
        self.end = end
        self.transition_prob = transition_prob
        self.reward = reward

class Action:
    def __init__(self, transitions):
        self.transitions = transitions


### Main

def main():
    t0 = time()

    ### Create states
    states = {}
    places = ["L1", "L2", "L3", "L4"]
    for i in places:
        states[i] = State(i)

    ### Create actions
    actions = {}
    actions["Pickup"] = Action(transitions=[
        Transition("L1", "L1", 0.7, 0),
        Transition("L1", "L2", 0.15, 19),
        Transition("L1", "L3", 0.09, 8.5),
        Transition("L1", "L4", 0.06, 7.75),
        Transition("L2", "L2", 0.5, 0),
        Transition("L2", "L1", 0.2, 9),
        Transition("L2", "L3", 0.3, 6.25),
        Transition("L3", "L3", 0.4, 0),
        Transition("L3", "L1", 0.18, 13.5),
        Transition("L3", "L4", 0.42, 7.8),
        Transition("L4", "L4", 0.2, 0),
        Transition("L4", "L1", 0.32, 9.75),
        Transition("L4", "L2", 0.48, 4),
    ])
    actions["MoveToL1"] = Action(transitions=[
        Transition("L2", "L1", 1, -1),
        Transition("L3", "L1", 1, -1.5),
        Transition("L4", "L1", 1, -1.25)
    ])
    actions["MoveToL2"] = Action(transitions=[
        Transition("L1", "L2", 1, -1),
        Transition("L3", "L2", 1, -1),
        Transition("L4", "L2", 1, -1)
    ])
    actions["MoveToL3"] = Action(transitions=[
        Transition("L1", "L3", 1, -1.5),
        Transition("L2", "L3", 1, -1.75),
        Transition("L4", "L3", 1, -1)
    ])
    actions["MoveToL4"] = Action(transitions=[
        Transition("L1", "L4", 1, -1.25),
        Transition("L3", "L4", 1, -1.2),
        Transition("L2", "L4", 1, -1)
    ])

    ### Value Iteration
    # Print initial values
    print(states["L1"].value)
    print(states["L2"].value)
    print(states["L3"].value)
    print(states["L4"].value)
    print("Initial state values" + "\n")

    i = 0
    while True:
        # Value iteration (estimate state values)
        for state in states.values():
            expected_utils = []
            for action_name, action in actions.items():
                expected_util = 0
                for t in action.transitions:
                    if t.start == state.pos:
                        state_new = states[t.end]
                        expected_util += t.transition_prob * (t.reward + (state_new.value_old * discount_factor))
                expected_utils.append((expected_util, action_name))
            best_action = max(expected_utils, key=lambda x: x[0])   # Action with max value / expected utility
            state.value, state.policy = best_action

        # Update state values
        for state in states.values():
            state.is_converged = abs(state.value_old - state.value) < convergence_tol
            state.value_old = state.value
        # Print
        i += 1
        print("state values L1", states["L1"].value, "action L1", states["L1"].policy)
        print("state values L2", states["L2"].value, "action L2", states["L1"].policy)
        print("state values L3", states["L3"].value, "action L4", states["L1"].policy)
        print("state values L4", states["L4"].value, "action L4", states["L1"].policy)
        print("Iteration: {0}".format(i) + "\n")
        # Convergence check
        is_converged = all([state.is_converged for state in states.values()])
        if is_converged: break

    print("Best policy" + "\n")
    print("Time taken is {:.4f} seconds".format(time() - t0))


if __name__ == "__main__":
    main()
