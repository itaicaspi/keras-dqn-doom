from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Reshape, Merge, Input, merge, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, Deconvolution2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.objectives import binary_crossentropy
from keras import backend as K
import scipy.ndimage
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from vizdoom import *
from main import *

def my_loss(x, x_decoded_mean):
    loss = K.mean(K.pow(x_decoded_mean-x, 2), axis=-1)
    return loss

class GAN(object):
    def __init__(self):
        self.frame_width = 80
        self.frame_height = 72
        self.autoencoder, self.state_encoder, self.state_decoder = self.autoencoder_model()
        self.generator, self.encoder, self.decoder = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.full_network  = self.full_network_model(self.generator, self.discriminator)
        self.predictor = self.predictor_model()
        self.state_encoder.load_weights('state_encoder_model_8000.h5')
        self.state_encoder.compile(Adam(lr=5e-4), "mse")
        self.autoencoder.load_weights('autoencoder_model_8000.h5')
        self.autoencoder.compile(Adam(lr=5e-4), "mse")
        self.vae, self.vae_encoder, self.vae_decoder = self.vae_conv_model()

    def vae_conv_model(self):
        # input image dimensions
        img_rows, img_cols, img_chns = 72, 80, 1
        # number of convolutional filters to use
        nb_filters = 64
        # convolution kernel size
        nb_conv = 3

        batch_size = 20
        if K.image_dim_ordering() == 'th':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)
        latent_dim = 2
        intermediate_dim = 128
        epsilon_std = 0.01
        nb_epoch = 5

        x = Input(batch_shape=(batch_size,) + original_img_size)
        conv_1 = Convolution2D(img_chns, 2, 2, border_mode='same', activation='relu')(x)
        conv_2 = Convolution2D(nb_filters, 2, 2,
                               border_mode='same', activation='relu',
                               subsample=(2, 2))(conv_1)
        conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                               border_mode='same', activation='relu',
                               subsample=(2, 2))(conv_2)
        conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                               border_mode='same', activation='relu',
                               subsample=(2, 2))(conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu')(flat)

        z_mean = Dense(latent_dim)(hidden)
        z_log_var = Dense(latent_dim)(hidden)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                      mean=0., std=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(nb_filters * 9 * 10, activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 9, 10)
        else:
            output_shape = (batch_size, 9, 10, nb_filters)

        decoder_reshape = Reshape(output_shape[1:])

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 18, 20)
        else:
            output_shape = (batch_size, 18, 20, nb_filters)

        decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                           output_shape,
                                           border_mode='same',
                                           subsample=(2, 2),
                                           activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 36, 40)
        else:
            output_shape = (batch_size, 36, 40, nb_filters)

        decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                           output_shape,
                                           border_mode='same',
                                           subsample=(2, 2),
                                           activation='relu')
        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 72, 80)
        else:
            output_shape = (batch_size, 72, 80, nb_filters)
        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2,
                                                  output_shape,
                                                  border_mode='valid',
                                                  subsample=(2, 2),
                                                  activation='relu')
        decoder_mean_squash = Convolution2D(img_chns, 2, 2,
                                            border_mode='same',
                                            activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        def vae_loss(x, x_decoded_mean):
            # NOTE: binary_crossentropy expects a batch_size by dim
            # for x and x_decoded_mean, so we MUST flatten these!
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = img_rows * img_cols * binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        vae = Model(x, x_decoded_mean_squash)
        vae.compile(optimizer='rmsprop', loss=vae_loss)
        vae.summary()

        # build a model to project inputs on the latent space
        encoder = Model(x, z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        generator = Model(decoder_input, _x_decoded_mean_squash)

        return vae, encoder, generator

    def vae_model(self):
        original_dim = 72*80
        latent_dim = 2
        intermediate_dim = 256
        nb_epoch = 50
        epsilon_std = 0.01

        input = Input(shape=(1,72,80))
        x = Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same')(input)
        x = ELU()(x)
        # x = BatchNormalization(mode=2)(x)
        x = Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = ELU()(x)
        # x = BatchNormalization(mode=2)(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = ELU()(x)
        x = Flatten()(x)
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(latent_dim,),
                                      mean=0., std=epsilon_std)
            return z_mean + K.exp(z_log_sigma) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        decoder_h = Dense(intermediate_dim, activation='relu')
        h_decoded = decoder_h(z)
        decoder_mean = Dense(original_dim, activation='sigmoid')
        x_decoded_mean = decoder_mean(h_decoded)

        # end-to-end autoencoder
        vae = Model(input, x_decoded_mean)

        # encoder, from inputs to latent space
        encoder = Model(input, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        def vae_loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        vae.compile(optimizer='rmsprop', loss=vae_loss)

        return vae, encoder, generator

    def predictor_model(self):
        input = Input(shape=(200,))

        x = Dense(200, activation='relu')(input)

        encoded_state = Dense(200, activation='relu')(x)

        # action encoder
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=8, activation='relu')(action)
        encoded_action = Dense(8, activation='relu')(x)

        x = merge([encoded_state, encoded_action], mode='concat')

        x = Dense(200)(x)

        x = ELU()(x)

        x = Dense(200)(x)

        next_state = ELU()(x)

        predictor = Model(input=[input, action], output=next_state)

        predictor.compile(optimizer=Adam(lr=5e-4), loss='mse')

        return predictor


    def autoencoder_model(self):
        a=1.0
        input_img = Input(shape=(4, self.frame_height, self.frame_width))

        # state encoder
        x = Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same')(input_img)
        x = ELU(a)(x)
        #x = BatchNormalization(mode=2)(x)
        x = Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = ELU(a)(x)
        #x = BatchNormalization(mode=2)(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = ELU(a)(x)
        #x = BatchNormalization(mode=2)(x)
        x = Flatten()(x)
        encoded_state = Dense(200)(x)
        encoded_state = ELU(a)(encoded_state)
        # encoded_state = Lambda(lambda a: K.greater(a, K.zeros_like(a)), output_shape=(32,))(encoded_state)
        state_encoder = Model(input=input_img, output=encoded_state)

        input = Input(shape=(200,))

        x1 = Dense(input_dim=200, output_dim=64 * 9 * 10)
        _x1 = x1(encoded_state)
        __x1 = x1(input)
        x2 = Reshape((64, 9, 10))
        _x2 = x2(_x1)
        __x2 = x2(__x1)
        x3 = ELU(a)
        _x3 = x3(_x2)
        __x3 = x3(__x2)
        x4 = Convolution2D(32, 3, 3, border_mode='same')
        _x4 = x4(_x3)
        __x4 = x4(__x3)
        x5 = UpSampling2D((2, 2))
        _x5 = x5(_x4)
        __x5 = x5(__x4)
        x6 = ELU(a)
        _x6 = x6(_x5)
        __x6 = x6(__x5)
        #x = BatchNormalization(mode=2)(x)
        x7 = Convolution2D(16, 3, 3, border_mode='same')
        _x7 = x7(_x6)
        __x7 = x7(__x6)
        x8 = UpSampling2D((2, 2))
        _x8 = x8(_x7)
        __x8 = x8(__x7)
        x9 = ELU(a)
        _x9 = x9(_x8)
        __x9 = x9(__x8)
        #x = BatchNormalization(mode=2)(x)
        x10 = Convolution2D(4, 3, 3, border_mode='same')
        _x10 = x10(_x9)
        __x10 = x10(__x9)
        x11 = UpSampling2D((2, 2))
        _x11 = x11(_x10)
        __x11 = x11(__x10)
        x12 = ELU(a)
        _x12 = x12(_x11)
        __x12 = x12(__x11)
        #x = BatchNormalization(mode=2)(x)
        decoded = Convolution2D(4, 3, 3, activation='sigmoid', border_mode='same')
        _decoded = decoded(_x12)
        __decoded = decoded(__x12)

        autoencoder = Model(input_img, _decoded)
        autoencoder.compile(optimizer=Adam(lr=5e-4), loss='mse')
        #autoencoder.summary()
        decoder = Model(input, __decoded)
        return autoencoder, state_encoder, decoder

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
        #encoded_state = Lambda(lambda a: K.greater(a, K.zeros_like(a)), output_shape=(32,))(encoded_state)
        state_encoder = Model(input=input_img, output=encoded_state)

        # action encoder
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=8, activation='relu')(action)
        encoded_action = Dense(8, activation='relu')(x)

        encoded = merge([encoded_state, encoded_action], mode='concat')
        #encoded = Lambda(lambda a: K.cast(a, 'float32'), output_shape=(40,))(encoded)

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
        #autoencoder.summary()

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


def padState(state, only_2d = False):
    # pad the inputs from the top and the bottom to from 120 to 128 to fit the network
    if only_2d:
        return np.lib.pad(state, ((6, 6), (0, 0)), 'constant', constant_values=(0))
    else:
        return np.lib.pad(state, ((0, 0), (6, 6), (0, 0)), 'constant', constant_values=(0))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val


def get_batch(full_curr = True, full_next = False, flatten = False):
    minibatch = np.array(agent.memory.sample_minibatch(batch_size, not_terminals=True))
    minibatch_transitions = minibatch[:, 1]
    if flatten:
        if full_curr:
            states = np.array(
                [padState(transition[0].preprocessed_curr[0]).flatten() for transition in minibatch_transitions]) / 255.0
        else:
            states = np.array(
                [padState(transition[0].preprocessed_curr[0][0], only_2d=True).flatten() for transition in
                 minibatch_transitions]) / 255.0
    else:

        if full_curr:
            states = np.array([padState(transition[0].preprocessed_curr[0]) for transition in minibatch_transitions]) / 255.0
        else:
            states = np.array([padState(np.expand_dims(transition[0].preprocessed_curr[0][0], 0)) for transition in
                                    minibatch_transitions]) / 255.0
            flattened_states = np.array(
                [padState(transition[0].preprocessed_curr[0][0], only_2d=True).flatten() for transition in
                 minibatch_transitions]) / 255.0

    actions = np.array([agent.environment.actions[transition[0].action] for transition in minibatch_transitions])
    if full_next:
        next_states = np.array([padState(transition[0].preprocessed_next[0]) for transition in minibatch_transitions]) / 255.0
    else:
        next_states = np.array([padState(np.expand_dims(transition[0].preprocessed_next[0][0], 0)) for transition in minibatch_transitions]) / 255.0
    #print(np.max(states[0][0]))
    #plt.imshow(states[0][0], cmap='Greys_r')
    #plt.show()
    return states, actions, next_states, flattened_states


if __name__ == "__main__":
    model = GAN()
    #print(model.encoder.input)
    #print(model.encoder.output)
    #print(model.generator.input)
    #print(model.generator.output)
    #print(model.discriminator.input)
    #print(model.discriminator.output)

    observe_episodes = 10
    train_episodes = 100000
    discriminator_pretrain_episodes = 0
    generator_pretrain_episodes = 0
    autoencoder_pretrain_episodes = 100000
    predictor_pretrain_episodes = 10000
    vae_pretrain_episodes = 100000
    steps_per_episode = 40
    batch_size = 20
    agent = Agent(algorithm=Algorithm.DDQN,
                  discount=0.99,
                  snapshot='',
                  max_memory=10000,
                  prioritized_experience=False,
                  exploration_policy=ExplorationPolicy.E_GREEDY,
                  learning_rate=2.5e-4,
                  level=Level.BASIC,
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
                  max_action_sequence_length = 1,
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
            next_state, reward, game_over = agent.step(action[0], action_idx[0])
            agent.store_next_state(next_state, reward, game_over, action_idx[0])
            steps += 1

    agent.environment.game.close()



    for i in range(vae_pretrain_episodes):
        print("vae pretraining episode " + str(i))
        states, actions, next_states, flattened_states = get_batch(False, False, False)
        loss = model.vae.train_on_batch(states, states)
        print("iteration " + str(i) + " loss: " + str(loss))
        generator_loss += [loss]

        if i % 2000 == 0:
            #x_test_encoded = model.vae_encoder.predict(states)
            #plt.figure(figsize=(6, 6))
            #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
            #plt.show()

            n = 15  # figure with 15x15 digits
            digit_size = 28
            width = 72
            height = 80
            figure = np.zeros((width * n, height * n))
            # we will sample n points within [-15, 15] standard deviations
            grid_x = np.linspace(-15, 15, n)
            grid_y = np.linspace(-15, 15, n)

            epsilon_std = 0.01

            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z_sample = np.array([[xi, yi]]) * epsilon_std
                    x_decoded = model.vae_decoder.predict(z_sample)
                    digit = x_decoded[0].reshape(width, height)
                    figure[i * width: (i + 1) * width,
                    j * height: (j + 1) * height] = digit

            plt.figure(figsize=(10, 10))
            plt.imshow(figure)
            plt.gray()
            plt.show()

            #for j in range(5):
            #    plt.subplot(1,5,j+1)
            #    #print(states[i][0].shape)
            #    plt.imshow(states[j][0])
            #    plt.gray()
            #plt.show()

    for i in range(predictor_pretrain_episodes):
        print("predictor pretraining episode " + str(i))
        states, actions, next_states = get_batch(True, True)
        encoded_curr = model.state_encoder.predict(states)
        encoded_next = model.state_encoder.predict(next_states)
        loss = model.predictor.train_on_batch([encoded_curr, actions], encoded_next)
        print("iteration " + str(i) + " loss: " + str(loss))
        generator_loss += [loss]

        if i % 1000 == 0:
            next = model.predictor.predict([encoded_curr, actions])
            print(encoded_next[0,:20])
            print(next[0,:20])
            output = model.state_decoder.predict(encoded_next)
            output_curr = model.state_decoder.predict(encoded_curr)

            ref = model.state_decoder.predict(next)
            for j in range(5):
                plt.subplot(5, 5, j + 1)
                # print(states[i][0].shape)
                plt.imshow(states[j][0])
                plt.gray()
                plt.subplot(5, 5, j + 6)
                # print(states[i][0].shape)
                plt.imshow(output_curr[j][0])
                plt.gray()
                plt.subplot(5, 5, j + 11)
                # print(states[i][0].shape)
                plt.imshow(next_states[j][0])
                plt.gray()
                plt.subplot(5, 5, j + 16)
                # print(output[i][0].shape)
                plt.imshow(output[j][0])
                plt.gray()
                plt.subplot(5, 5, j + 21)
                # print(output[i][0].shape)
                plt.imshow(ref[j][0])
                plt.gray()
            plt.show()
            snapshot = 'predictor_model_' + str(i) + '.h5'
            #print(" >> saving snapshot to " + snapshot)
            model.predictor.save_weights(snapshot, overwrite=True)


    for i in range(autoencoder_pretrain_episodes):
        print("autoencoder pretraining episode " + str(i))
        states, actions, next_states = get_batch()
        loss = model.autoencoder.train_on_batch(states, states)
        print("iteration " + str(i) + " loss: " + str(loss))
        generator_loss += [loss]
        if i % 2000 == 0 and i != 0:
            output = model.autoencoder.predict(states)
            """
            test_predicted_next, code = model.predict_next_state(states, actions)
            print(test_predicted_next.shape)
            scipy.misc.toimage(np.reshape(next_states,(20*72, 80)), cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_true.jpg')
            scipy.misc.toimage(np.reshape(test_predicted_next,(20*72,80)), cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_fake.jpg')
            scipy.misc.toimage(code, cmin=0.0, cmax=1.0).save(
                'results/pretraining_' + str(int(i / 10)) + '_code.jpg')

            predicted_next = model.predict_multiple_action_single_state(states[0])
            print(np.array(predicted_next).shape)
            """
            for j in range(5):
                plt.subplot(2,5,j+1)
                #print(states[i][0].shape)
                plt.imshow(states[j][0])
                plt.gray()
                plt.subplot(2,5,j+6)
                #print(output[i][0].shape)
                plt.imshow(output[j][0])
                plt.gray()
            #scipy.misc.toimage(predicted_next[i][0][0], cmin=0.0, cmax=1.0).save(
            #    'results/pretraining_action_' + str(int(i)) + '.jpg')

            plt.show()
            snapshot = 'autoencoder_model_' + str(i) + '.h5'
            #print(" >> saving snapshot to " + snapshot)
            model.autoencoder.save_weights(snapshot, overwrite=True)

            snapshot = 'state_encoder_model_' + str(i) + '.h5'
            # print(" >> saving snapshot to " + snapshot)
            model.state_encoder.save_weights(snapshot, overwrite=True)

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
