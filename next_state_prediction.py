from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Reshape, Merge, Input, merge, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
import scipy.ndimage
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from vizdoom import *
from main import Agent

class GAN(object):
    def __init__(self):
        self.generator     = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.full_network  = self.full_network_model(self.generator, self.discriminator)

    # GENERATOR (input: [state, action], output: [fake_next_state])

    # expected input: [action: (3), state: (4,128,160)]
    def generator_model(self):

        # state encoder
        state = Input(shape=(4, 128, 160))
        x = Convolution2D(16, 5, 5, subsample=(2, 2), input_shape=(4, 128, 160), border_mode='same', init='glorot_normal')(state)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(500, init='glorot_normal')(x)
        state_embedding = Activation('sigmoid')(x)

        # action encoder
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=20, init='glorot_normal')(action)
        x = Activation('relu')(x)
        x = Dense(100, init='glorot_normal')(x)
        action_embedding = Activation('sigmoid')(x)

        # encoder
        embedding = merge([state_embedding, action_embedding], mode='concat')

        # genrerator
        x = Dense(input_dim=600, output_dim=128 * 8 * 10, init='glorot_normal')(embedding)
        x = Reshape((128, 8, 10))(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(64, 5, 5, border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(32, 5, 5, border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(16, 5, 5, border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(1, 5, 5, border_mode='same', init='glorot_normal')(x)
        x = BatchNormalization(mode=2)(x)
        next_state = Activation('sigmoid')(x)
        model = Model(input=[state, action], output=[next_state])
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
        #model.summary()
        return model

    # DISCRIMINATOR (input: [curr_state + next_state, action], output: classification)

    # expected input: [action: (3), state_stack: (5,128,160)]
    def discriminator_model(self):
        # state discriminator
        state_stack = Input(shape=(5, 128, 160))
        x = Convolution2D(16, 5, 5, subsample=(4, 4), input_shape=(5, 128, 160), border_mode='same', init='glorot_normal')(state_stack)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Convolution2D(32, 5, 5, subsample=(4, 4), border_mode='same', init='glorot_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Convolution2D(64, 5, 5, subsample=(4, 4), border_mode='same', init='glorot_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(200, init='glorot_normal')(x)
        state_discriminator = Activation('sigmoid')(x)

        # action discriminator
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=20, init='glorot_normal')(action)
        x = Activation('relu')(x)
        x = Dense(100, init='glorot_normal')(x)
        action_discriminator = Activation('sigmoid')(x)

        # discriminator
        x = merge([state_discriminator, action_discriminator], mode='concat')
        prediction = Dense(2, activation='softmax', init='glorot_normal')(x)
        model = Model(input=[state_stack, action], output=[prediction])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5))
        #model.summary()
        return model

    # expected input: [state_stack: (4,128,160), [action: (3), state_stack: (4,128,160)]]
    def full_network_model(self, generator, discriminator):
        state = Input(shape=(4, 128, 160))
        action = Input(shape=(3,))

        make_trainable(discriminator, False)

        G = generator([state, action])
        state_stack = merge([state, G], mode='concat', concat_axis=1)
        prediction = discriminator([state_stack, action])

        model = Model(input=[state, action], output=[prediction])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
        model.summary()
        return model

    def predict_next_state(self, state, action):
        predicted_next_state = self.generator.predict([state, action])
        return predicted_next_state

    def discriminate_next_state(self, state, action, next_state):
        state_stack = np.concatenate((state, next_state), axis=0)
        prediction = self.discriminator.predict([state_stack, action])
        return prediction

    def train_discriminator(self, states, actions, fake_next_states, true_next_states):
        # [1,0] - fake, [0,1] - true
        make_trainable(self.discriminator, True)
        fake_state_stacks = np.concatenate((states, fake_next_states), axis=1)
        true_state_stacks = np.concatenate((states, true_next_states), axis=1)

        state_stacks = np.concatenate((fake_state_stacks, true_state_stacks), axis=0)
        targets = np.concatenate((np.tile([1,0],(len(actions),1)), np.tile([0,1],(len(actions),1))), axis=0)
        actions = np.concatenate((actions, actions), axis=0)

        # shuffle the order
        perm = np.random.permutation(len(actions))
        state_stacks = state_stacks[perm]
        targets = targets[perm]
        actions = actions[perm]

        loss = self.discriminator.train_on_batch([state_stacks, actions], targets)
        return loss

    def train_generator(self, states, actions):
        make_trainable(self.discriminator, False)
        targets = np.tile([1, 0], (len(actions), 1))
        loss = self.full_network.train_on_batch([states, actions], targets)
        return loss


def padState(state):
    # pad the inputs from the top and the bottom to from 120 to 128 to fit the network
    return np.lib.pad(state, ((0, 0), (4, 4), (0, 0)), 'constant', constant_values=(0))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val


def get_batch():
    minibatch = np.array(agent.memory.sample_minibatch(batch_size, not_terminals=True))
    minibatch_transitions = minibatch[:, 1]
    states = np.array([padState(transition.preprocessed_curr[0]) for transition in minibatch_transitions])
    actions = np.array([agent.environment.actions[transition.action] for transition in minibatch_transitions])
    next_states = np.array([padState(np.expand_dims(transition.preprocessed_next[0][0], 0)) for transition in minibatch_transitions])
    return states, actions, next_states


if __name__ == "__main__":
    model = GAN()
    #print(model.encoder.input)
    #print(model.encoder.output)
    #print(model.generator.input)
    #print(model.generator.output)
    #print(model.discriminator.input)
    #print(model.discriminator.output)

    observe_episodes = 500
    train_episodes = 10000
    discriminator_pretrain_episodes = 0
    generator_pretrain_episodes = 100000
    steps_per_episode = 40
    batch_size = 20
    agent = Agent(discount=0.99, snapshot='', max_memory=50000, prioritized_experience=False)

    generator_loss = []
    discriminator_loss = []


    # observation
    for i in range(observe_episodes):
        print("observe episode " + str(i))
        agent.environment.new_episode()
        steps = 0
        curr_return = 0
        loss = 0
        game_over = False
        while not game_over and steps < steps_per_episode:
            _, _, game_over = agent.step()
            steps += 1

    agent.environment.game.close()

    # pretraining
    for i in range(discriminator_pretrain_episodes):
        print("discriminator pretraining episode " + str(i))
        states, actions, next_states = get_batch()
        predicted_next = model.predict_next_state(states, actions)
        loss = model.train_discriminator(states, actions, predicted_next, next_states)
        print("iteration " + str(i) + " loss: " + str(loss))
        discriminator_loss += [loss]

    for i in range(generator_pretrain_episodes):
        print("generator pretraining episode " + str(i))
        states, actions, next_states = get_batch()
        loss = model.generator.train_on_batch([states, actions], next_states)
        print("iteration " + str(i) + " loss: " + str(loss))
        generator_loss += [loss]

        if i % 20 == 0:
            test_predicted_next = model.predict_next_state(states, actions)
            scipy.misc.toimage(next_states[0][0], cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_true.jpg')
            scipy.misc.toimage(test_predicted_next[0][0], cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_fake.jpg')

            snapshot = 'gan_model.h5'
            print(" >> saving snapshot to " + snapshot)
            model.generator.save_weights(snapshot, overwrite=True)

    plt.plot(range(len(discriminator_loss)), discriminator_loss, 'g', range(len(generator_loss)), generator_loss, 'r')
    plt.show()

    test_states, test_actions, test_next_states = get_batch()

    # training
    for i in range(train_episodes):
        if i % 5 == 0:
            test_predicted_next = model.predict_next_state(test_states, test_actions)
            scipy.misc.toimage(test_predicted_next[0][0], cmin=0.0, cmax=1.0).save(
                'results/outfile' + str(int(i / 10)) + '.jpg')
            fake_state_stacks = np.concatenate((test_states, test_predicted_next), axis=1)
            true_state_stacks = np.concatenate((test_states, test_next_states), axis=1)
            state_stacks = np.concatenate((fake_state_stacks, true_state_stacks), axis=0)
            targets = np.concatenate((np.tile([1, 0], (len(test_actions), 1)), np.tile([0, 1], (len(test_actions), 1))),
                                     axis=0)
            actions = np.concatenate((test_actions, test_actions), axis=0)
            predictions = model.discriminator.predict([state_stacks, actions])

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / float(len(targets))
            print("Accuracy: " + str(accuracy))
            # plt.imshow(test_predicted_next[0][0], cmap='Greys_r')
            # plt.show()

        print(">> training episode " + str(i))

        states, actions, next_states = get_batch()
        predicted_next = model.predict_next_state(states, actions)
        loss = model.train_discriminator(states, actions, predicted_next, next_states)
        discriminator_loss += [loss]
        print("train discriminator. loss: " + str(loss))

        #states, actions, next_states = get_batch()
        #for j in range(1):
        loss = model.train_generator(states, actions)
        generator_loss += [loss]
        print("train generator. loss: " + str(loss))


    plt.plot(range(len(discriminator_loss)), discriminator_loss, 'g', range(len(generator_loss)), generator_loss, 'r')
    plt.show()
