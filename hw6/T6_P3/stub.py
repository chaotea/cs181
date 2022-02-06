# Imports.
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pygame as pg

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self, epsilon, alpha, gamma):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.iterations = 0

        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        # initialize our Q-value grid that has an entry for each action and state
        # (action, rel_x, rel_y, gravity)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE, 2))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.iterations = 0

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state['tree']['dist']) // X_BINSIZE)
        rel_y = int((state['tree']['top'] - state['monkey']['top']) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # 0a. If this is the first state, don't jump
        if not self.last_state:
            self.last_action = 0
            self.last_state = state
            return self.last_action

        # Ob. Determine the gravity
        if self.last_action == 0:
            if self.last_state['monkey']['vel'] - state['monkey']['vel'] == 1:
                self.gravity = 0  # if gravity is 1
            else:
                self.gravity = 1  # if gravity is 4

        # 1. Discretize 'state' to get your transformed 'current state' features
        cur_x, cur_y = self.discretize_state(state)
        last_x, last_y = self.discretize_state(self.last_state)

        # 2. Perform the Q-Learning update using 'current state' and the 'last state'
        prev = self.Q[self.last_action, last_x, last_y, self.gravity]
        update = prev + self.alpha * (self.last_reward + self.gamma * np.max(self.Q[:, cur_x, cur_y, self.gravity]) - prev)
        self.Q[self.last_action, last_x, last_y, self.gravity] = update

        # 3. Choose the next action using an epsilon-greedy policy with epsilon decay after 50 iterations
        if npr.rand() < self.epsilon and self.iterations < 50:
            new_action = npr.choice([0, 1])
        else:
            new_action = np.argmax(self.Q[:, cur_x, cur_y, self.gravity])

        # 4. Update last action and state
        new_state = state
        self.last_action = new_action
        self.last_state = new_state

        self.iterations += 1
        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Define hyperparameters.
    epsilon = 0.001  # exploration rate
    alpha = 0.1  # learning rate
    gamma = 0.9  #discount factor

    # Select agent.
    agent = Learner(epsilon=epsilon, alpha=alpha, gamma=gamma)

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)

    # Print scores and frequency of scores over 50
    print(f'scores: {hist}')
    print(f'scores over 50: {np.sum(np.array(hist) > 50)}%')

    # Graph scores
    plt.style.use('seaborn')
    plt.figure()
    plt.title(f'epsilon = {epsilon}, alpha = {alpha}, gamma = {gamma}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.scatter(np.arange(100), hist, s=20)
    plt.savefig(f'plots/epsilon = {epsilon}, alpha = {alpha}, gamma = {gamma}.png')
    plt.show()

    # Save history. 
    np.save('hist', np.array(hist))
