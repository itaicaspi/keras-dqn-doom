import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten
from keras.optimizers import adam
from keras.initializations import uniform
from vizdoom import *
import scipy.ndimage
from time import sleep
import matplotlib.pyplot as plt

class Environment(object):
    def __init__(self):
        #self.actions = ['Left', 'Right', 'Shoot']
        self.actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.game = DoomGame()
        self.game.load_config("basic.cfg")
        self.game.init()
        self.screen_width = self.game.get_screen_width()
        self.screen_height = self.game.get_screen_height()

    def step(self, action):
        reward = self.game.make_action(action)
        next_state = self.game.get_state().image_buffer
        game_over = self.game.is_episode_finished()
        return next_state, reward, game_over

    def get_curr_state(self):
        return self.game.get_state().image_buffer

    def new_episode(self):
        self.game.new_episode()

    def is_game_over(self):
        return self.game.is_episode_finished()


class Agent(object):
    def __init__(self, discount):
        self.discount = discount
        self.epsilon_annealing = 2e4
        self.epsilon_start = 0.7
        self.epsilon_end = 0.05
        self.epsilon = self.epsilon_start
        self.environment = Environment()
        self.memory = ExperienceReplay()
        self.preprocessed_curr = []
        self.win_count = 0
        self.batch_size = 10
        self.state_stack = 4
        # model
        self.learning_rate = 1e-4
        self.model = Sequential()
        self.model.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu', input_shape=(self.state_stack,120,160), init='uniform'))
        self.model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        self.model.add(Flatten())
        self.model.add(Dense(len(self.environment.actions),init='uniform'))
        self.model.compile(adam(lr=self.learning_rate), "mse")

    def preprocess(self, state):
        return scipy.misc.imresize(state[0], 0.5)

    def get_inputs_and_targets(self, minibatch):
        targets = list()
        inputs = list()
        for transition, game_over in minibatch:
            inputs.append(transition.preprocessed_curr[0])
            target = self.model.predict(transition.preprocessed_curr)[0]
            if game_over:
                target[transition.action] = transition.reward
            else:
                Q_sa = self.model.predict(transition.preprocessed_next)
                target[transition.action] = transition.reward + self.discount * np.max(Q_sa)
            #print(target)
            targets.append(target)
        return np.array(inputs), np.array(targets)

    def step(self):
        # if no current state is present, create one by stacking the duplicated current state
        if self.preprocessed_curr == []:
            self.preprocessed_curr = list()
            sub_state = self.preprocess(self.environment.get_curr_state())
            for t in range(self.state_stack):
                self.preprocessed_curr.append(sub_state)
            self.preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.state_stack, 120, 160))

        # choose action
        Q = self.model.predict(self.preprocessed_curr, batch_size=1)
        coin_toss = np.random.rand(1)[0]
        if coin_toss > self.epsilon:
            action_idx = np.argmax(Q)
            action = self.environment.actions[action_idx]
        else:
            action_idx = np.random.randint(len(self.environment.actions))
            action = self.environment.actions[action_idx]

        # anneal epsilon value
        if self.epsilon > self.epsilon_end:
            self.epsilon -= 1/float(self.epsilon_annealing)

        # repeat action several times and stack the states
        reward = 0
        game_over = False
        preprocessed_next = list()
        for t in range(self.state_stack):
            s, r, game_over = self.environment.step(action)
            reward += r # reward is accumulated
            if game_over:
                break
            preprocessed_next.append(self.preprocess(s))

        # episode finished
        if not game_over:
            preprocessed_next = np.reshape(preprocessed_next, (1, self.state_stack, 120, 160))
        else:
            preprocessed_next = []
            self.environment.new_episode()

        if reward > 0 and game_over:
            self.win_count += 1

        # store transition
        self.memory.remember(Transition(self.preprocessed_curr, action_idx, reward, preprocessed_next), game_over)

        self.preprocessed_curr = preprocessed_next

        return reward, np.max(Q), game_over

    def train(self):
        minibatch = self.memory.sample_minibatch(self.batch_size)
        inputs, targets = self.get_inputs_and_targets(minibatch)
        return self.model.train_on_batch(inputs, targets)


class Transition(object):
    def __init__(self, preprocessed_curr, action, reward, preprocessed_next):
        self.preprocessed_curr = preprocessed_curr
        self.action = action
        self.reward = reward
        self.preprocessed_next = preprocessed_next


class ExperienceReplay(object):
    def __init__(self, max_memory=50000):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def sample_minibatch(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        indices = np.random.choice(len(self.memory), batch_size)
        minibatch = list()
        for idx in indices:
            minibatch.append(self.memory[idx])
        return minibatch


episodes = 1000000
steps_per_episode = 40
agent = Agent(discount=0.99)
total_steps = 0
average_return = 0
returns = []
Qs = []
for i in range(episodes):
    agent.environment.new_episode()
    steps = 0
    curr_return = 0
    loss = 0
    game_over = False
    while not game_over and steps < steps_per_episode:
        reward, mean_Q, game_over = agent.step()
        steps += 1
        curr_return += reward
        Qs += [mean_Q]
        if i > 30:
            loss += agent.train()

    average_return = 0.99*average_return+0.01*curr_return
    total_steps += steps


    print("steps = " + str(total_steps) + " epsilon = " + str(agent.epsilon))
    print("episode = " + str(i) + " wins = " + str(agent.win_count) + " loss = " + str(loss))
    print("current_return = " + str(curr_return) + " average return = " + str(average_return))
    returns += [average_return]

    if i % 10000 == 0:
        plt.plot(range(len(returns)), returns)
        #plt.ion()
        plt.show()

        plt.plot(range(len(Qs)), Qs)
        plt.show()

agent.environment.game.close()