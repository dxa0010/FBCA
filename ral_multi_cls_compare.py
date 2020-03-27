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
import envs.ral_multi_class_env_test_adalt
import envs.ral_multi_class_env_test_car
import envs.ral_multi_class_env_test_white
import envs.ral_multi_class_env_test_red
import envs.ral_multi_class_env_test_google
import envs.ral_multi_class_env_test_trip

args = sys.argv

ENV_NAME = "ralenv_multi_classify_1-v0"

ENV_NAME_TEST_1 = 'ralenv_multi_classify_adalt-v0'
ENV_NAME_TEST_2 = 'ralenv_multi_classify_car-v0'
ENV_NAME_TEST_3 = 'ralenv_multi_classify_red-v0'
ENV_NAME_TEST_4 = 'ralenv_multi_classify_white-v0'
ENV_NAME_TEST_5 = 'ralenv_multi_classify_google-v0'
ENV_NAME_TEST_6 = 'ralenv_multi_classify_trip-v0'

env = gym.make(ENV_NAME_TEST_1)
env_1 = gym.make(ENV_NAME_TEST_1)
env_2 = gym.make(ENV_NAME_TEST_2)
env_3 = gym.make(ENV_NAME_TEST_3)
env_4 = gym.make(ENV_NAME_TEST_4)
env_5 = gym.make(ENV_NAME_TEST_5)
env_6 = gym.make(ENV_NAME_TEST_6)

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

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights(os.path.join("model","dqn_multi_cls_{}_weights.h5f".format(ENV_NAME)))

env_1.weight_num = args[1]
dqn.test(env_1, nb_episodes=5, visualize=True)

# env_2.seed(123)
env_2.weight_num = args[1]
dqn.test(env_2, nb_episodes=5, visualize=True)

# env_3.seed(123)
env_3.weight_num = args[1]
dqn.test(env_3, nb_episodes=5, visualize=True)

# env_4.seed(123)
env_4.weight_num = args[1]
dqn.test(env_4, nb_episodes=5, visualize=True)

env_5.weight_num = args[1]
dqn.test(env_5, nb_episodes=5, visualize=True)

# env_4.seed(123)
env_6.weight_num = args[1]
dqn.test(env_6, nb_episodes=5, visualize=True)