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
import envs.ral_multi_class_env_test_5
import envs.ral_multi_class_env_test_10
import envs.ral_multi_class_env_test_15
import envs.ral_multi_class_env_test_25


args = sys.argv


ENV_NAME_TEST_1 = 'ralenv_multi_classify_test_1-v0'
ENV_NAME_TEST_2 = 'ralenv_multi_classify_test_2-v0'

# ENV_NAME_TEST = 'ralenv_multi_classify_test-v0'
# ENV_NAME_TEST_RANDOM = 'ralenv_multi_classify_random-v0'

env = gym.make(ENV_NAME_TEST_1)
env_1 = gym.make(ENV_NAME_TEST_1)
env_2 = gym.make(ENV_NAME_TEST_2)

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
# history = dqn.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=1000)
# dqn.save_weights(os.path.join("model","dqn_multi_cls_{}_weights.h5f".format(ENV_NAME)), overwrite=True)
dqn.load_weights(os.path.join("model","dqn_multi_cls_ralenv_multi_classify_{}-v0_weights.h5f".format(args[1])))

# env_1.seed(123)
env_1.weight_num = args[1]
dqn.test(env_1, nb_episodes=5, visualize=True)

# env_2.seed(123)
env_2.weight_num = args[1]
dqn.test(env_2, nb_episodes=5, visualize=True)