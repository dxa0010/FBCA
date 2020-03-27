import numpy as np
import pandas as pd
import keras 
import sys
import gym
import os
import xgboost as xgb
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from keras import backend as K
import envs.ral_multi_class_env_learn_adult
if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0,1"
set_session(tf.Session(config=config))

ENV_NAME = 'ralenv_multi_classify_adult_learn-v0'

env = gym.make(ENV_NAME)
# env_test = gym.make(ENV_NAME_TEST)
# env_test_random = gym.make(ENV_NAME_TEST_RANDOM)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape[0]
print(nb_actions)
print(env.observation_space.shape)
input_obs = Input(shape=(1,env.observation_space.shape[0]))
x = Flatten()(input_obs)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(nb_actions)(x)
x = Activation('linear')(x)

model = Model(inputs = input_obs, outputs = x)

# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights(os.path.join("model","dqn_multi_cls_ralenv_multi_classify_{}-v0_weights.h5f".format(1)))

history = dqn.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=1000)
dqn.save_weights(os.path.join("model","dqn_multi_class_1_plus_adult.h5f"), overwrite=True)
# dqn.load_weights("dqn_multi_cls_{}_weights.h5f".format("ral_env"))

# env.seed(123)
# dqn.test(env_test, nb_episodes=5, visualize=True)

# env.seed(123)
# dqn.test(env_test_random, nb_episodes=5, visualize=True)
