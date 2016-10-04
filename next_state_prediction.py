from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Reshape, Merge, Input, merge, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras import backend as K
import scipy.ndimage
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from vizdoom import *
from main import *

class GAN(object):
    def __init__(self):
        self.frame_width = 80
        self.frame_height = 72
        self.generator, self.encoder, self.decoder = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.full_network  = self.full_network_model(self.generator, self.discriminator)

    # GENERATOR (input: [state, action], output: [fake_next_state])

    # expected input: [action: (3), state: (4,128,160)]
    def generator_model(self):
        input_img = Input(shape=(4, self.frame_height, self.frame_width))

        # state encoder
        x = Convolution2D(16, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(input_img)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(x)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(x)
        x = BatchNormalization(mode=2)(x)
        x = Flatten()(x)
        encoded_state = Dense(32, activation='relu')(x)
        encoded_state = Lambda(lambda a: K.greater(a, K.zeros_like(a)), output_shape=(32,))(encoded_state)
        state_encoder = Model(input=input_img, output=encoded_state)

        # action encoder
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=8, activation='relu')(action)
        encoded_action = Dense(8, activation='relu')(x)

        encoded = merge([encoded_state, encoded_action], mode='concat')
        encoded = Lambda(lambda a: K.cast(a, 'float32'), output_shape=(40,))(encoded)

        x = Dense(input_dim=40, output_dim=64 * 9 * 10)(encoded)
        x = Reshape((64, 9, 10))(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model([input_img, action], decoded)
        autoencoder.compile(optimizer=Adam(lr=5e-4), loss='binary_crossentropy')
        autoencoder.summary()

        #######################################################
        encoded_state = Input(shape=(32,))
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=8, activation='relu')(action)
        encoded_action = Dense(8, activation='relu')(x)
        encoded = merge([encoded_state, encoded_action], mode='concat')
        encoded = Lambda(lambda a: K.cast(a, 'float32'), output_shape=(40,))(encoded)
        x = Dense(input_dim=40, output_dim=64 * 9 * 10)(encoded)
        x = Reshape((64, 9, 10))(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(mode=2)(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
        decoder = Model([encoded_state, action], decoded)
        decoder.compile(optimizer=Adam(lr=5e-4), loss='binary_crossentropy')

        return autoencoder, state_encoder, decoder

    # DISCRIMINATOR (input: [curr_state + next_state, action], output: classification)

    # expected input: [action: (3), state_stack: (5,128,160)]
    def discriminator_model(self):
        # state discriminator
        state_stack = Input(shape=(5, self.frame_height, self.frame_width))
        x = Convolution2D(16, 5, 5, subsample=(4, 4), input_shape=(5, self.frame_height, self.frame_width), border_mode='same', init='glorot_normal')(state_stack)
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
        state = Input(shape=(4, self.frame_height, self.frame_width))
        action = Input(shape=(3,))

        make_trainable(discriminator, False)

        G = generator([state, action])
        state_stack = merge([state, G], mode='concat', concat_axis=1)
        prediction = discriminator([state_stack, action])

        model = Model(input=[state, action], output=[prediction])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
        #model.summary()
        return model

    def predict_multiple_action_single_state(self, state):
        predicted_next_states = []
        for action in range(3):
            one_hot = [False] * 3
            one_hot[action] = True
            predicted_next_state = self.generator.predict([np.expand_dims(state, 0), np.expand_dims(one_hot,0)])
            predicted_next_states += [predicted_next_state]

        return predicted_next_states

    def predict_next_state(self, state, action):
        predicted_next_state = self.generator.predict([state, action])
        encoded_state = self.encoder.predict(state)
        return predicted_next_state, encoded_state

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
    return np.lib.pad(state, ((0, 0), (6, 6), (0, 0)), 'constant', constant_values=(0))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val


def get_batch():
    minibatch = np.array(agent.memory.sample_minibatch(batch_size, not_terminals=True))
    minibatch_transitions = minibatch[:, 1]
    states = np.array([padState(transition.preprocessed_curr[0]) for transition in minibatch_transitions]) / 255.0
    actions = np.array([agent.environment.actions[transition.action] for transition in minibatch_transitions])
    next_states = np.array([padState(np.expand_dims(transition.preprocessed_next[0][0], 0)) for transition in minibatch_transitions]) / 255.0
    #print(np.max(states[0][0]))
    #plt.imshow(states[0][0], cmap='Greys_r')
    #plt.show()
    return states, actions, next_states


if __name__ == "__main__":
    model = GAN()
    #print(model.encoder.input)
    #print(model.encoder.output)
    #print(model.generator.input)
    #print(model.generator.output)
    #print(model.discriminator.input)
    #print(model.discriminator.output)

    observe_episodes = 20
    train_episodes = 10000
    discriminator_pretrain_episodes = 0
    generator_pretrain_episodes = 100000
    steps_per_episode = 40
    batch_size = 20
    agent = Agent(algorithm=Algorithm.DDQN,
                  discount=0.99,
                  snapshot='',
                  max_memory=10000,
                  prioritized_experience=False,
                  exploration_policy=ExplorationPolicy.E_GREEDY,
                  learning_rate=2.5e-4,
                  level=Level.HEALTH,
                  history_length=4,
                  batch_size=10,
                  temperature=1,
                  combine_actions=True  ,
                  train=True,
                  skipped_frames=4,
                  target_update_freq=1000,
                  epsilon_start=0.7,
                  epsilon_end=0.1,
                  epsilon_annealing_steps=5e4,
                  architecture=Architecture.DUELING,
                  visible=False)

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
            action, action_idx, mean_Q = agent.predict()
            next_state, reward, game_over = agent.step(action, action_idx)
            agent.store_next_state(next_state, reward, game_over, action_idx)
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

        if i % 200 == 0:
            """
            test_predicted_next, code = model.predict_next_state(states, actions)
            print(test_predicted_next.shape)
            scipy.misc.toimage(np.reshape(next_states,(20*72, 80)), cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_true.jpg')
            scipy.misc.toimage(np.reshape(test_predicted_next,(20*72,80)), cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_fake.jpg')
            scipy.misc.toimage(code, cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_code.jpg')
            """
            predicted_next = model.predict_multiple_action_single_state(states[0])
            print(np.array(predicted_next).shape)
            for i in range(3):
                #plt.subplot(1,3,i+1)
                #plt.imshow(predicted_next[i][0][0])
                #plt.gray()
                scipy.misc.toimage(predicted_next[i][0][0], cmin=0.0, cmax=1.0).save(
                    'results/pretraining_action_' + str(int(i)) + '.jpg')

            #plt.show()
            snapshot = 'gan_model_' + str(i) + '.h5'
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
