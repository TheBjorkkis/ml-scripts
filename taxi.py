
import gym
import random
import numpy as np
import time

# Muodostetaan ympäristö.
env = gym.make("Taxi-v3")

# Täytetään taulukko tuloksia varten.
#-1000 määritellään, koska mahdollisimman pieni
next_state = -1000*np.ones((501,6))
next_reward = -1000*np.ones((501,6))

# Alustetaan koulutusta varten tilat
# actions = mihin voi mennä
# states = mitkä tilat on ympärillä
# qtable = tulos
actions = env.action_space.n
states = env.observation_space.n
qtable = np.zeros((states, actions))

# Koulutusta varten alustetut arvot
all_episodes = 50000
tot_train_ep = 100
max_steps = 99      # max steps per ep
learning_rate = 0.7
gamma = 0.618       # discount rate

eps = 1.0           # exploration rate
max_eps = 1.0       # expl probability at start
min_eps = 0.01      # minimun expl prob
decay_rate = 0.01

# Randomilla pelataan läpi all_episodes verran
for episode in range(all_episodes):

    # Nollataan kaikki tilat välissä
    state = env.reset()
    step = 0 # Liikkeiden määrä
    done = False #Pääsikö määränpäähän
    
    #Montako liikettä saa maksimissaan tehdä
    for step in range(max_steps):

        # choose an action in the current world state
        # we need  random value
        eps_tradeoff = random.uniform(0,1)
        
        # Googleta, en muista enää tätä
        if eps_tradeoff > eps:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        
        # Tarkista tilan tiedot, missä kohdassa taxi on asettunut
        new_state, reward, done, info = env.step(action)
        
        # Päivitä qtableen tiedot kehityksestä (esitetty tätä kaavaa luennolla)
        qtable[state, action] = qtable[state,action] + learning_rate * (reward + gamma
              * np.max(qtable[new_state,:]) - qtable[state, action])
        
        # Tallennetaan uudet tiedot ylös
        state = new_state

        #Jos määränpää on saavutettu, niin lopetetaan pelaaminen tältä erää
        if done == True:
            break
        
    #YKsi peli pelattu lisää ja kaava (kvg)
    episode += 1
    eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*episode)
        
            
            
# Pelataan samantyylisellä logiikalla kuin koulutusvaiheessa
test_tot_reward = 0
test_tot_actions = 0
past_observation = -1
observation = env.reset();
for t in range(50):
    test_tot_actions = test_tot_actions+1
    action = np.argmax(qtable[observation])
    if (observation == past_observation):
        # Kuulema tällä estetään jumittuminen samaan paikkaan
        action = random.sample(range(0,6),1)
        action = action[0]
    past_observation = observation
    observation, reward, done, info = env.step(action)
    test_tot_reward = test_tot_reward+reward
    env.render()
    time.sleep(1)
    if done:
        break
print("Total reward: ")
print(test_tot_reward)
print("Total actions: ")
print(test_tot_actions)
