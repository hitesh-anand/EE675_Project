import numpy as np
from scipy.optimize import linprog


class WolfPHCPlayer:

    def __init__(self, numStates, numActionsA, numActionsB, decay, expl=0.2, gamma=1, lr_l=1e-2, lr_w=1e-3):
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = .5
        # self.V = np.ones(numStates)
        self.Q = np.ones((numStates, numActionsA))
        self.pi = np.ones((numStates, numActionsA)) / numActionsA
        self.pi_avg = self.pi   # average policy
        
        self.numStates = numStates
        self.C = {i:0 for i in range(numStates)}    # state counter
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB
        self.num_actions = self.numActionsA
        self.learning = True
        self.lr_l = lr_l    # losing learning rate
        self.lr_w = lr_w    # winning learning rate

    def chooseAction(self, state, restrict=None):
        '''
        epsilon greedy action selection
        '''
        epsilon = self.expl
        p = np.random.choice(2, 1, [epsilon, 1-epsilon])
        if p == 0:
            # choose random action
            a = np.random.choice(self.num_actions, 1, p=[1/self.num_actions]*self.num_actions)
        else:
            # print(self.pi[state])
            a = np.random.choice(self.num_actions, 1, p=self.pi[state])
        return a[0]

    def getReward(self, initialState, finalState, actions, reward, restrictActions=None):
        '''
        Updates for WoLF-PHC iteration
        '''
        
        if not self.learning:
            return
        

        actionA, actionB = actions
        # update Q value
        self.Q[initialState][actionA] += self.alpha * (reward +self.gamma * self.Q[finalState].max() - self.Q[initialState][actionA])

        # update state counter
        self.C[initialState] += 1

        mu = {}
        delta = {}
        
        s = initialState
        a = actionA

        # set d and delta 
        r1 = sum([self.pi[s][a] * self.Q[s][a] for a in range(self.num_actions)])
        r2 = sum([self.pi_avg[s][a] * self.Q[s][a] for a in range(self.num_actions)])

        d = self.lr_l
        if r1 > r2:
            d = self.lr_w
    

        for i in range(self.num_actions):
            mu[i] = min(self.pi[s][i], d/(self.num_actions - 1))
        
 
        # set delta
        for i in range(self.num_actions):
            delta[i] = -mu[i]
            if i == np.argmax(self.Q[s]):
                # print('----------------HERE--------P---------')
                delta[i] += sum(list(mu.values()))

        # update pi and pi_avg (policy and average policy)
        for i in range(self.num_actions):
            self.pi_avg[s][i] += 1/self.C[s] * (self.pi[s][i] - self.pi_avg[s][i])
            self.pi[s][i] += delta[i]


    def policyForState(self, state):
        for i in range(self.numActionsA):
            print("Actions %d : %f" % (i, self.pi[state, i]))