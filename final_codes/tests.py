import numpy as np
import matplotlib.pyplot as plt
from q_player import QPlayer
from random_player import RandomPlayer
from minimaxq_player import MinimaxQPlayer
from soccer import Soccer
from wolfphc_player import WolfPHCPlayer
import sys

class Tester:

    def __init__(self, game, playerA=None, playerB=None):
        self.game = game
        self.playerA = playerA
        self.playerB = playerB

    def resultToReward(self, result, actionA=None, actionB=None):
        if result >= 0:
            reward = (result*(-2) + 1)
        else:
            reward = 0
        return reward

    def restrictActions(self):
        return [None, None]

    def plotPolicy(self, player):
        for state in range(player.numStates):
            print("\n=================")
            self.game.draw(*self.stateToBoard(state))
            # print("State value: %s" % player.V[state])
            print(player.Q[state])
            player.policyForState(state)

    def plotResult(self, wins):
        lenWins = len(wins)
        sumWins = (wins == [[0], [1], [-2]]).sum(1)
        print("Wins A : %d (%0.1f%%)" % (sumWins[0], (100. * sumWins[0] / lenWins)))
        print("Wins B : %d (%0.1f%%)" % (sumWins[1], (100. * sumWins[1] / lenWins)))
        print("Draws  : %d (%0.1f%%)" % (sumWins[2], (100. * sumWins[2] / lenWins)))

        plt.plot((wins == 0).cumsum())
        plt.plot((wins == 1).cumsum())
        plt.xlabel('Episodes')
        plt.ylabel('No of wins')
        plt.legend(('WinsA', 'WinsB'), loc=(0.6, 0.8))
        plt.show()


class SoccerTester(Tester):

    def __init__(self, game):
        Tester.__init__(self, game)

    def boardToState(self):
        game = self.game
        xA, yA = game.positions[0]
        xB, yB = game.positions[1]
        sA = yA * game.w + xA
        sB = yB * game.w + xB
        sB -= 1 if sB > sA else 0
        state = (sA * (game.w * game.h - 1) + sB) + (game.w * game.h) * (game.w * game.h - 1) * game.ballOwner
        return state

    def stateToBoard(self, state):
        game = self.game
        ballOwner = state / ((game.w * game.h) * (game.w * game.h - 1))
        state = state % ((game.w * game.h) * (game.w * game.h - 1))

        sA = state / (game.w * game.h - 1)
        sB = state % (game.w * game.h - 1)
        sB += 1 if sB >= sA else 0

        xA = sA % game.w
        yA = sA / game.w
        xB = sB % game.w
        yB = sB / game.w

        return [[[xA, yA], [xB, yB]], ballOwner]

    def resultToReward(self, result, actionA=None, actionB=None):
        factor = 1
        return Tester.resultToReward(self, result) * factor


def testGame(playerA, playerB, gameTester, iterations):
    wins = np.zeros(iterations)

    for i in np.arange(iterations):
        if (i % (iterations / 10) == 0):
            print("%d%%" % (i * 100 / iterations))
        gameTester.game.restart()
        result = -1
        while result == -1:
            state = gameTester.boardToState()
            restrictA, restrictB = gameTester.restrictActions()
            actionA = playerA.chooseAction(state, restrictA)
            actionB = playerB.chooseAction(state, restrictB)
            result = gameTester.game.play(actionA, actionB)
            reward = gameTester.resultToReward(result, actionA, actionB)
            newState = gameTester.boardToState()

            playerA.getReward(state, newState, [actionA, actionB], reward, [restrictA, restrictB])
            playerB.getReward(state, newState, [actionB, actionA], -reward, [restrictB, restrictA])

        wins[i] = result
    return wins


def testSoccer(iterations, policy='wolfphc'):
    boardH = 4
    boardW = 5
    numStates = (boardW * boardH) * (boardW * boardH - 1) * 2
    numActions = 5
    drawProbability = 0.001
    decay = 10**(-2. / iterations * 0.05)

    ### CHOOSE PLAYER_A TYPE
    # playerA = RandomPlayer(numActions)
    if policy == 'wolfphc':
        print('Initialized Player A with WoLF-PHC')
        playerA = WolfPHCPlayer(numStates, numActions, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    else:
        print('Initialized Player A with Minimax Q learning')
        playerA = MinimaxQPlayer(numStates, numActions, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    #  playerA = WolfPHCPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerA = np.load('SavedPlayers/minimaxQ_SoccerA_100000.npy', allow_pickle=True).item()

    ### CHOOSE PLAYER_B TYPE
    print('Initialized Player B with random policy')
    playerB = RandomPlayer(numActions)
    # playerB = QPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playrB = MinimaxWolfPHCPlayer(numStates, numActions, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerB = WolfPHCPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerB = WolfPHCPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerB = np.load('SavedPlayers/Q_SoccerB_100000.npy').item()

    ### INSTANTIATE GAME AND TESTER
    game = Soccer(boardH, boardW, drawProbability=drawProbability)
    tester = SoccerTester(game)

    ### RUN TEST
    wins = testGame(playerA, playerB, tester, iterations)

    ### DISPLAY RESULTS
    tester.plotPolicy(playerA)
    # tester.plotPolicy(playerB)
    tester.plotResult(wins)
    print('----------------------------testSoccer finished----------------------------')

    np.save("SoccerA_10000", playerA)
    np.save("SoccerB_10000", playerB)


if __name__ == '__main__':

    policy = 'wolfphc'
    if(len(sys.argv) > 1):
        if(sys.argv[1] == 'minimaxql'):
            policy = 'minimaxql'
            print('-----------------Running Minimax Q learning vs Random---------------------')
        else:
            print('-------------------Running WoLF PHC vs Random--------------------------------')
    else:
        print('-------------------Running WoLF PHC vs Random--------------------------------')

    



    ### RUN TESTS
    testSoccer(5000, policy)
    ### RUN PERFORMANCE TESTS
    # testSoccerPerformance()

    ### TO PROFILE ALGORITHM TIMING PERFORMANCE
    # import cProfile
    # cProfile.run('testSoccer(1000)')
    # cProfile.run('testOshiZumo(1000)')
