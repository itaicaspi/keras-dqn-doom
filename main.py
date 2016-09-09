import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, Flatten
from keras.optimizers import adam
from keras.initializations import uniform
from vizdoom import *
import scipy.ndimage
from time import sleep
import matplotlib.pyplot as plt
import itertools as it
import datetime
from enum import Enum


image_height, image_width = 240, 320

class Level(Enum):
    BASIC = "configs/basic.cfg"
    HEALTH = "configs/health_gathering.cfg"
    DEATHMATCH = "configs/deathmatch.cfg"
    DEFEND = "configs/defend_the_center.cfg"

class Algorithm(Enum):
    DQN = 1
    DDQN = 2

class ExplorationPolicy(Enum):
    E_GREEDY = 1
    SOFTMAX = 2
    SHIFTED_MULTINOMIAL = 3

class Environment(object):
    def __init__(self, level = Level.BASIC, combine_actions = False):
        self.game = DoomGame()
        self.game.load_config(level.value)
        self.game.init()
        self.actions_num = self.game.get_available_buttons_size()
        self.combine_actions = combine_actions
        self.actions = []
        if self.combine_actions:
            for perm in it.product([False, True], repeat=self.actions_num):
                self.actions.append(list(perm))
        else:
            for action in range(self.actions_num):
                one_hot = [False] * self.actions_num
                one_hot[action] = True
                self.actions.append(one_hot)
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
    def __init__(self, discount, level, algorithm, prioritized_experience, max_memory, exploration_policy,
                 learning_rate, state_stack, batch_size, combine_actions, temperature=10, snapshot=''):
        """

        :param discount:
        :param algorithm: either DQN or DDQN
        :param prioritized_experience: prioritize experience replay according to temporal difference
        :param snapshot:
        :param max_memory:
        :param policy: either multinomial or e-greedy
        """
        # e-greedy policy
        self.epsilon_annealing = 3e4 #steps
        self.epsilon_start = 0.7
        self.epsilon_end = 0.05
        self.epsilon = self.epsilon_start

        # softmax / multinomial policy
        self.average_minimum = 0 # for multinomial policy
        self.temperature = temperature

        self.policy = exploration_policy

        # initialization
        self.environment = Environment(level=level, combine_actions=combine_actions)
        self.memory = ExperienceReplay(max_memory=max_memory, prioritized=prioritized_experience)
        self.preprocessed_curr = []
        self.win_count = 0
        self.curr_step = 0

        self.state_width = 320
        self.state_height = 240
        self.scale = self.environment.screen_width / float(self.state_width)

        # training
        self.discount = discount
        self.state_stack = state_stack
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = 500

        self.algorithm = algorithm
        if snapshot != '':
            print("loading snapshot " + str(snapshot))
            self.target_network = load_model(snapshot)
            self.online_network = load_model(snapshot)
        else:
            self.target_network = self.create_network()
            self.online_network = self.create_network()


    def create_network(self):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu', input_shape=(self.state_stack,image_height, image_width), init='uniform'))
        model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', init='uniform'))
        model.add(Flatten())
        model.add(Dense(len(self.environment.actions),init='uniform'))
        model.compile(adam(lr=self.learning_rate), "mse")
        return model

    def preprocess(self, state):
        # resize image and convert to greyscale
        if self.scale == 1:
            return np.mean(state,0)
        else:
            return scipy.misc.imresize(np.mean(state,0), self.scale)

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
                if self.algorithm == Algorithm.DQN:
                    Q_sa = self.target_network.predict(transition.preprocessed_next)
                    TD_error = transition.reward + self.discount * np.max(Q_sa) - target[transition.action]
                elif self.algorithm == Algorithm.DDQN:
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

    def softmax_selection(self, Q):
        # compute thresholds and choose a random number
        exp_Q = np.array(np.exp(Q/float(self.temperature)), copy=True)
        prob = np.random.rand(1)
        importances = [action_value/float(np.sum(exp_Q)) for action_value in exp_Q]
        thresholds = np.cumsum(importances)
        # multinomial sampling according to priorities
        for action_idx, threshold in zip(range(len(thresholds)), thresholds):
            if prob < threshold:
                action = self.environment.actions[action_idx]
                return action, action_idx
        return self.environment.actions[len(exp_Q)-1], len(exp_Q)-1

    def shifted_multinomial_selection(self, Q):
        # Q values are shifted so that we won't have negative values
        self.average_minimum = 0.95 * self.average_minimum + 0.05 * np.min(Q)
        shifted_Q = np.array(Q - min(self.average_minimum, np.min(Q)), copy=True)
        # compute thresholds and choose a random number
        prob = np.random.rand(1)
        importances = [action_value/float(np.sum(shifted_Q)) for action_value in shifted_Q]
        thresholds = np.cumsum(importances)
        # multinomial sampling according to priorities
        for action_idx, threshold in zip(range(len(thresholds)), thresholds):
            if prob < threshold:
                action = self.environment.actions[action_idx]
                return action, action_idx

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
            curr_state = self.environment.get_curr_state()
            sub_state = self.preprocess(self.environment.get_curr_state())
            for t in range(self.state_stack):
                self.preprocessed_curr.append(sub_state)
            self.preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.state_stack, image_height, image_width))

        # choose action
        Q = self.online_network.predict(self.preprocessed_curr, batch_size=1)

        if self.policy == ExplorationPolicy.E_GREEDY:
            action, action_idx = self.e_greedy(Q)
        elif self.policy == ExplorationPolicy.SHIFTED_MULTINOMIAL:
            action, action_idx = self.shifted_multinomial_selection(Q)
        elif self.policy == ExplorationPolicy.SOFTMAX:
            action, action_idx = self.softmax_selection(Q)
        else:
            exit()

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
            preprocessed_next = np.reshape(preprocessed_next, (1, self.state_stack, image_height, image_width))
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
        self.beta = self.beta_end
        self.sum_powered_priorities = 0 # sum p^alpha
        self.memory = list()

    def remember(self, transition, game_over):
        # set the priority to the maximum current priority
        transition_powered_priority = 1e-7 ** self.alpha
        if self.memory != []:
            transition_powered_priority = np.max(self.memory,0)[2]
        self.sum_powered_priorities += transition_powered_priority
        # store transition and clean up memory if necessary
        self.memory.append([transition, game_over, transition_powered_priority])
        if len(self.memory) > self.max_memory:
            self.sum_powered_priorities -= self.memory[0][2]
            del self.memory[0]

    def sample_minibatch(self, batch_size, not_terminals=False):
        batch_size = min(len(self.memory), batch_size)
        if self.prioritized:
            # prioritized experience replay
            probs = np.random.rand(batch_size)
            importances = [self.get_transition_importance(idx) for idx in range(len(self.memory))]
            thresholds = np.cumsum(importances)
            # multinomial sampling according to priorities
            indices = []
            for p in probs:
                for idx, threshold in zip(range(len(thresholds)), thresholds):
                    if p < threshold:
                        indices += [idx]
                        break
        else:
            indices = np.random.choice(len(self.memory), batch_size)
        minibatch = list()
        for idx in indices:
            while not_terminals and self.memory[idx][1] == True:
                idx = np.random.choice(len(self.memory), 1)[0]
            weight = 0
            if self.prioritized:
                weight = self.get_transition_weight(idx)
            minibatch.append([idx] + self.memory[idx][0:2] + [weight]) # idx, transition, game_over, weight

        if self.prioritized:
            max_weight = np.max(minibatch,0)[3]
            for idx in range(len(minibatch)):
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
        weight = 1/float(self.get_transition_importance(transition_idx)*self.max_memory)**self.beta
        return weight


def run_training(args):

    agent = Agent(algorithm=args["algorithm"],
                  discount=args["discount"],
                  snapshot=args["snapshot"],
                  max_memory=args["max_memory"],
                  prioritized_experience=args["prioritized_experience"],
                  exploration_policy=args["exploration_policy"],
                  learning_rate=args["learning_rate"],
                  level=args["level"],
                  state_stack=args["state_stack"],
                  batch_size=args["batch_size"],
                  temperature=args["temperature"],
                  combine_actions=args["combine_actions"])

    # initialize
    total_steps = 0
    average_return = 0
    average_Q = 0
    returns = []
    Qs = []
    for i in range(args["episodes"]):
        agent.environment.new_episode()
        steps = 0
        curr_return = 0
        curr_Qs = 0
        loss = 0
        game_over = False
        while not game_over and steps < args["steps_per_episode"]:
            reward, mean_Q, game_over = agent.step()
            steps += 1
            curr_return += reward
            curr_Qs += mean_Q

            if i > args["start_learning_after"] and args["train"] == True:
                loss += agent.train()

        n = float(args["average_over_num_episodes"])
        average_Q = (1-1/n) * average_Q + (1/n) * curr_Qs/float(steps)
        average_return = (1-1/n) * average_return + (1/n) * curr_return
        total_steps += steps

        print("")
        print(str(datetime.datetime.now()))
        print("episode = " + str(i) + " steps = " + str(total_steps))
        print("epsilon = " + str(agent.epsilon) + " loss = " + str(loss))
        print("current_return = " + str(curr_return) + " average return = " + str(average_return))

        returns += [average_return]
        Qs += [average_Q]

        if i % args["snapshot_episodes"] == args["snapshot_episodes"] - 1:
            snapshot = 'model_' + str(i + 1) + '.h5'
            print(str(datetime.datetime.now()) + " >> saving snapshot to " + snapshot)
            agent.target_network.save(snapshot, overwrite=True)

    agent.environment.game.close()
    return returns, Qs


if __name__ == "__main__":
    softmax = {
        "snapshot_episodes": 1000,
        "episodes": 2000,
        "steps_per_episode": 40, # 4300 for deathmatch, 300 for health gathering
        "average_over_num_episodes": 50,
        "start_learning_after": 5,
        "algorithm": Algorithm.DDQN,
        "discount": 0.99,
        "max_memory": 10000,
        "prioritized_experience": False,
        "exploration_policy": ExplorationPolicy.SOFTMAX,
        "learning_rate": 2.5e-4,
        "level": Level.DEFEND,
        "combine_actions": True,
        "temperature": 10,
        "batch_size": 10,
        "state_stack": 4,
        "snapshot": 'model_1.h5',
        "train": False
    }
    multinomial = softmax.copy()
    multinomial["exploration_policy"] = ExplorationPolicy.SHIFTED_MULTINOMIAL

    egreedy = softmax.copy()
    egreedy["exploration_policy"] = ExplorationPolicy.E_GREEDY

    runs = [softmax, multinomial, egreedy]

    colors = ["r", "g", "b"]
    for color, run in zip(colors, runs):
        # run agent
        returns, Qs = run_training(run)

        # plot results
        plt.figure(1)
        plt.plot(range(len(returns)), returns, color)
        plt.xlabel("episode")
        plt.ylabel("average return")
        plt.title("Average Return")

        plt.figure(2)
        plt.plot(range(len(Qs)), Qs, color)
        plt.xlabel("episode")
        plt.ylabel("mean Q value")
        plt.title("Mean Q Value")

    plt.show()
