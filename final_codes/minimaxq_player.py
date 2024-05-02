import numpy as np
from scipy.optimize import linprog


class MinimaxQPlayer:

    def __init__(self, numStates, numActionsA, numActionsB, decay, expl, gamma):
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 1
        self.V = np.ones(numStates)
        self.Q = np.ones((numStates, numActionsA, numActionsB))
        self.pi = np.ones((numStates, numActionsA)) / numActionsA
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB
        self.num_actions = numActionsA
        self.learning = True


    def chooseAction(self, state, restrict=None):
        '''
            Given the current state, choose action for the agent
            as per the epsilon-greedy policy
        '''
        epsilon = self.expl
        p = np.random.choice(2, 1, [epsilon, 1-epsilon])
        if p == 0:
            # choose random action
            a = np.random.choice(self.num_actions, 1, p=[1/self.num_actions]*self.num_actions)
        else:
           
            a = np.random.choice(self.num_actions, 1, p=self.pi[state])
        return a[0]

        

    

    def getReward(self, initialState, finalState, actions, reward, restrictActions=None):
        '''
            Given the current and next states, actions taken and the reward, update the 
            Q and Value functions for the agent
        '''
        if self.learning:
            actionA, actionB = actions
            self.Q[initialState, actionA, actionB] += self.alpha * (reward + self.gamma * self.V[finalState] - self.Q[initialState, actionA, actionB])
            self.V[initialState] = self.updatePolicy(initialState)
            self.alpha *= self.decay

    def updatePolicy(self, state, retry=False):
        '''
            Given the current state, implements Linear Programming solution for
            updating the Value function for the current state
        '''
        c = np.zeros(self.numActionsA + 1)
        c[0] = -1
        A_ub = np.ones((self.numActionsB, self.numActionsA + 1))
        A_ub[:, 1:] = -self.Q[state].T
        b_ub = np.zeros(self.numActionsB)
        A_eq = np.ones((1, self.numActionsA + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.numActionsA

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        # If solved successfully, update the policy
        if res.success and np.all(res.x[1:] >= 0):
            self.pi[state] = res.x[1:]
        elif not retry:
            return self.updatePolicy(state, retry=True)
        else:
            print("Alert : %s" % res.message)
            return self.V[state]

        return res.x[0]

    def policyForState(self, state):
        for i in range(self.numActionsA):
            print("Actions %d : %f" % (i, self.pi[state, i]))


