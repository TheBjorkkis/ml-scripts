import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median


LR = 1e-3  # Learningrate
env = gym.make('CartPole-v1')
env.reset()

goal_steps = 500
score_requirements = 50  # Minimum score of games
initial_games = 10000


def initial_population():
    """
    Initial_population() fills the scoretable with the observations and rewards.
    Every step is taken randomly and loop breaks if score is reached or enviroment reaches done state
    Then after filling the table we transform steps into two possible steps (left and right).
    """

    print("Filling the table")

    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):

        score = 0
        game_memory = []
        prev_observation = []
        done = False
        env.reset()

        for _ in range(goal_steps):

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:

                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirements:

            accepted_scores.append(score)

            for data in game_memory:

                if data[1] == 1:

                    output = [0, 1]

                elif data[1] == 0:

                    output = [1, 0]

                training_data.append([data[0], output])

        scores.append(score)

    env.close()

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    return training_data


def neural_network(input_size):
    """
    Untrained neural_network is built so it can be trained in train_model() function
    """
    
    print("Creating the network")

    network = input_data(shape=[None, input_size, 1], name="input")

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    """
        Training data is divided into two vectors. 
        One is X (inputs) and the another one is Y (targets).
        If there isn't already a neural network to train, an empty network will be built.
        Otherwise network will be fit with these two vectors and is returned.
    """

    print("Training the model")

    X = np.array([i[0] for i in training_data]
                 ).reshape(-1, len(training_data[0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': Y}, n_epoch=4,
              snapshot_step=500, show_metric=True, run_id='AI_test_results')

    return model


def main():

    training_data = initial_population()
    model = train_model(training_data)

    print("Let the games begin...")

    scores = []
    choices = []

    for each_game in range(10):

        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        done = False

        for _ in range(goal_steps):
            env.render()

            if len(prev_obs) == 0:
                action = env.action_space.sample()

            else:
                action = np.argmax(model.predict(
                    prev_obs.reshape(-1, len(prev_obs), 1))[0])
            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward

            if done:
                break

        scores.append(score)
    print("------------------------------------------------")
    print('Average Score', sum(scores) / len(scores))
    print('Choice 1: {}, Choice 2: {}'.format(choices.count(
        1) / len(choices), choices.count(0) / len(choices)))
    print("------------------------------------------------")
    env.close()

main()
