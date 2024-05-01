from soccer import soccer_env
from neural_net import Q_network
import random
import numpy as np

def epsilon_greedy(Q_network,state,epsilon):
    if random.random() < epsilon:
        return random.randint(0,5)
    else:
        Q_values = []
        for i in range(6):
            state_action = np.concatenate((state,i),axis=0)
            Q_value = Q_network.online_network.forward(state_action)
            Q_values.append((Q_value,i))
        Q_values.sort(reverse=True)
        return Q_values[0][1]

def main():
    env = soccer_env.SoccerEnv()
    player_A = Q_network()
    player_B = Q_network()
    num_episodes = 1000
    epochs = 1000
    frequency = 50
    epsilon=0.1
    stochastic_param=20
    gamma=0.9
    fill_memory = 100
    for episode in num_episodes:
        current_state_A,current_state_B,BallOwner = env.reset()
        for epoch in range(epochs):
            # for agent a
            action_A = epsilon_greedy(player_A,np.concatenate((current_state_A,BallOwner),axis=0),epsilon)
            action_B = epsilon_greedy(player_B,np.concatenate((current_state_B,BallOwner),axis=0),epsilon)
            next_state_A,next_state_B,next_BallOwner,reward_A,reward_B,done_env = env.move(action_A,action_B)
            player_A.replay_memory.append((np.concatenate((current_state_A,current_state_B,BallOwner),axis=0),action_A,reward_A,np.concatenate((next_state_A,next_BallOwner),axis=0),done_env))
            player_B.replay_memory.append((np.concatenate((current_state_A,current_state_B,BallOwner),axis=0),action_B,reward_B,np.concatenate((next_state_B,next_BallOwner),axis=0),done_env))
            current_state_A = next_state_A
            current_state_B = next_state_B
            if epoch < fill_memory and episode == 0:
                continue
            else:
                sample_for_A = random.sample(list(player_A.replay_memory), stochastic_param)
                sample_for_B = random.sample(list(player_B.replay_memory), stochastic_param)
                input_ls=[]
                label_ls=[]
                for sample in sample_for_A:
                    state,action,reward,next_state,done = sample
                    if done:
                        target = reward
                    else:
                        target = reward + gamma*max([player_A.target_network.forward(np.concatenate((next_state,i),axis=0)) for i in range(6)])
                    input_ls.append(np.concatenate((state,action),axis=0))
                    label_ls.append(target)
                player_A.online_network.train(input_ls,label_ls)
                input_ls=[]
                label_ls=[]
                for sample in sample_for_B:
                    state,action,reward,next_state,done = sample
                    if done:
                        target = reward
                    else:
                        target = reward + gamma*max([player_B.target_network.forward(np.concatenate((next_state,i),axis=0)) for i in range(6)])
                    input_ls.append(np.concatenate((state,action),axis=0))
                    label_ls.append(target)
                player_B.online_network.train(input_ls,label_ls)
            if epoch%frequency==0:
                player_A.target_network.load_state_dict(player_A.online_network.state_dict())
                player_B.target_network.load_state_dict(player_B.online_network.state_dict())
            if done_env:
                break 