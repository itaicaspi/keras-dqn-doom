import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten
from keras.optimizers import adam
from keras.initializations import uniform
from vizdoom import *
import scipy.ndimage
from time import sleep
import matplotlib.pyplot as plt
import itertools as it
import datetime

class Environment(object):
    def __init__(self):
        self.game = DoomGame()
        #self.game.load_config("basic.cfg")
        self.game.load_config("health_gathering.cfg")
        self.game.init()
        self.actions_num = self.game.get_available_buttons_size()
        self.actions = []
        for perm in it.product([False, True], repeat=self.actions_num):
            self.actions.append(list(perm))
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
    def __init__(self, discount, algorithm='DDQN', prioritized_experience=False, snapshot=''):
        # e-greedy policy
        self.epsilon_annealing = 3e4 #steps
        self.epsilon_start = 0.7
        self.epsilon_end = 0.05
        self.epsilon = self.epsilon_start

        # initialization
        self.environment = Environment()
        self.memory = ExperienceReplay(prioritized=prioritized_experience)
        self.preprocessed_curr = []
        self.win_count = 0
        self.curr_step = 0

        # training
        self.discount = discount
        self.state_stack = 4
        self.learning_rate = 1e-4
        self.batch_size = 10
        self.target_update_freq = 500
        self.target_network = self.create_network()
        self.online_network = self.create_network()
        self.algorithm = algorithm
        if snapshot != '':
            self.target_network.load_weights(snapshot)
            self.online_network.load_weights(snapshot)

    def create_network(self):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu', input_shape=(self.state_stack,120,160), init='uniform'))
        model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        model.add(Flatten())
        model.add(Dense(len(self.environment.actions),init='uniform'))
        model.compile(adam(lr=self.learning_rate), "mse")
        return model

    def preprocess(self, state):
        return scipy.misc.imresize(state[0], 0.5)

    def get_inputs_and_targets(self, minibatch):
        targets = list()
        inputs = list()
        samples_weights = list()
        for idx, transition, game_over, sample_weight in minibatch:
            TD_error = 0 # temporal difference error (actual action-value - predicted action value)
            inputs.append(transition.preprocessed_curr[0])

            # get the current action-values
            target = self.online_network.predict(transition.preprocessed_curr)[0]
            if game_over:
                target[transition.action] = transition.reward
            else:
                if self.algorithm == 'DQN':
                    Q_sa = self.target_network.predict(transition.preprocessed_next)
                    TD_error = transition.reward + self.discount * np.max(Q_sa) - target[transition.action]
                elif self.algorithm == 'DDQN':
                    best_next_action = np.argmax(self.online_network.predict(transition.preprocessed_next))
                    Q_sa = self.target_network.predict(transition.preprocessed_next)[0][best_next_action]
                    TD_error = transition.reward + self.discount * Q_sa - target[transition.action]
                target[transition.action] += TD_error
            targets.append(target)

            # updates priority and weight for prioritized experience replay
            if self.memory.prioritized:
                self.memory.update_transition_priority(idx, np.abs(TD_error))
                samples_weights.append(sample_weight)

        return np.array(inputs), np.array(targets), np.array(samples_weights)

    def e_greedy(self, Q):
        # choose action randomly or greedily
        coin_toss = np.random.rand(1)[0]
        if coin_toss > self.epsilon:
            action_idx = np.argmax(Q)
        else:
            action_idx = np.random.randint(len(self.environment.actions))
        action = self.environment.actions[action_idx]

        # anneal epsilon value
        if self.epsilon > self.epsilon_end:
            self.epsilon -= float(self.epsilon_start - self.epsilon_end)/float(self.epsilon_annealing)

        return action, action_idx

    def step(self):
        # if no current state is present, create one by stacking the duplicated current state
        if self.preprocessed_curr == []:
            self.preprocessed_curr = list()
            sub_state = self.preprocess(self.environment.get_curr_state())
            for t in range(self.state_stack):
                self.preprocessed_curr.append(sub_state)
            self.preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.state_stack, 120, 160))

        # choose action
        Q = self.online_network.predict(self.preprocessed_curr, batch_size=1)
        action, action_idx = self.e_greedy(Q)

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
        self.curr_step += 1

        # copy online network to target network
        if self.curr_step % self.target_update_freq == 0:
            self.target_network.set_weights(self.online_network.get_weights())

        return reward, np.mean(Q), game_over

    def train(self):
        minibatch = self.memory.sample_minibatch(self.batch_size)
        inputs, targets, samples_weights = self.get_inputs_and_targets(minibatch)
        if self.memory.prioritized:
            return self.online_network.train_on_batch(inputs, targets, sample_weight=samples_weights)
        else:
            return self.online_network.train_on_batch(inputs, targets)


class Transition(object):
    def __init__(self, preprocessed_curr, action, reward, preprocessed_next):
        self.preprocessed_curr = preprocessed_curr
        self.action = action
        self.reward = reward
        self.preprocessed_next = preprocessed_next


class ExperienceReplay(object):
    # memory consists of tuples [transition, game_over, priority^alpha]
    def __init__(self, max_memory=50000, prioritized=False):
        self.max_memory = max_memory
        self.prioritized = prioritized
        self.alpha = 0.6 # prioritization factor
        self.beta_start = 0.4
        self.beta_end = 1
        self.beta = self.beta0
        self.sum_powered_priorities = 0 # sum p^alpha
        self.memory = list()

    def remember(self, transition, game_over):
        # set the priority to the maximum current priority
        transition_priority = 0
        if self.memory != []:
            transition_priority = np.max(self.memory,0)[2]
        # store transition and clean up memory if necessary
        self.memory.append([transition, game_over, transition_priority])
        if len(self.memory) > self.max_memory:
            self.sum_powered_priorities -= self.memory[0][2]
            del self.memory[0]

    def sample_minibatch(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        if self.prioritized:
            # prioritized experience replay
            probs = np.random.rand(batch_size)
            importances = [self.get_transition_importance(idx) for idx in range(len(self.memory))]
            thresholds = np.cumsum(importances)

            indices = []
            for p in probs:
                for idx, threshold in zip(range(thresholds), thresholds):
                    if p < threshold:
                        indices += [idx]
                        break
        else:
            indices = np.random.choice(len(self.memory), batch_size)
        minibatch = list()
        for idx in indices:
            weight = 0
            if self.prioritized:
                weight = self.get_transition_weight(idx)
            minibatch.append([idx] + self.memory[idx][0:1] + [weight]) # idx, transition, game_over, weight
        if self.prioritized:
            max_weight = np.max(minibatch,0)[3]
            for idx in indices:
                minibatch[idx][3] /= float(max_weight) # normalize weights relative to the minibatch
        return minibatch

    def update_transition_priority(self, transition_idx, priority):
        self.sum_powered_priorities -= self.memory[transition_idx][2]
        powered_priority = (priority+np.spacing(0)) ** self.alpha
        self.sum_powered_priorities += powered_priority
        self.memory[transition_idx][2] = powered_priority

    def get_transition_importance(self, transition_idx):
        powered_priority = self.memory[transition_idx][2]
        importance = powered_priority / float(self.sum_powered_priorities)
        return importance

    def get_transition_weight(self, transition_idx):
        weight = 1/(self.get_transition_importance(transition_idx)*self.max_memory)**self.beta
        return weight


# params
plot_results_episodes = 100
episodes = 1000000
steps_per_episode = 300
average_over_num_episodes = 100
agent = Agent(discount=0.99, snapshot='')

# initialize
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

    average_return = (1-1/float(average_over_num_episodes))*average_return+(1/float(average_over_num_episodes))*curr_return
    total_steps += steps


    print(str(datetime.datetime.now()))
    print("episode = " + str(i) + " steps = " + str(total_steps))
    print("epsilon = " + str(agent.epsilon) + " loss = " + str(loss))
    print("wins = " + str(agent.win_count) + " current_return = " + str(curr_return) + " average return = " + str(average_return))

    returns += [average_return]

    if i % plot_results_episodes == plot_results_episodes-1:
        snapshot = 'dqn_model_' + str(i+1) + '.h5'
        print(str(datetime.datetime.now()) + " >> saving snapshot to " + snapshot)
        agent.target_network.save_weights(snapshot, overwrite=True)

        # plt.ion()
        plt.plot(range(len(returns)), returns)
        plt.show()

        plt.plot(range(len(Qs)), Qs)
        plt.show()

agent.environment.game.close()
