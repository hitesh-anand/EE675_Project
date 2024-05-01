"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import random
import operator
import numpy as np
import pygame
import sys
import os

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 128, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 60
HEIGHT = 60

# This sets the margin between each cell
MARGIN = 1


class Soccer(object):

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0,0)]

    def __init__(self, args, current_path):
# h=4, w=5, pA=[3, 2], pB=[1, 1], goalPositions=[1, 2], ballOwner=0, drawProbability=0
        self.h = 4
        self.w = 5
        self.goalPositions = [1,2]
        self.positions = np.array([[3,2],[1,1]])
        self.initPositions = np.array([[3,2],[1,1]])
        self.ballOwner = 0
        self.drawProbability = 0.1

        self.num_agents = args['agents_number']

        # self.num_preys = 1
        # self.preys_mode = args['preys_mode']
        # self.num_walls = 0
        # self.grid_size = args['grid_size']
        self.game_mode = args['game_mode']
        self.reward_mode = args['reward_mode']
        self.state_size = 1

        # self.state_size = (self.num_preys + self.num_predators + self.num_walls)*2
        # self.predators_positions = []
        # self.preys_positions = []
        # self.walls_positions = []
        self.render_flag = args['render']
        self.recorder_flag = args['recorder']
        # enables visualizer
        if self.render_flag:
            [self.screen, self.my_font] = self.gui_setup()
            self.step_num = 1

            resource_path = os.path.join(current_path, 'environments')  # The resource folder path
            resource_path = os.path.join(resource_path, 'predators_prey')  # The resource folder path
            image_path = os.path.join(resource_path, 'images')  # The image folder path

            img = pygame.image.load(os.path.join(image_path, 'predator_prey.jpg')).convert()
            self.img_predator_prey = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'predator.jpg')).convert()
            self.img_predator = pygame.transform.scale(img, (WIDTH, WIDTH))
            img = pygame.image.load(os.path.join(image_path, 'prey.jpg')).convert()
            self.img_prey = pygame.transform.scale(img, (WIDTH, WIDTH))

            if self.recorder_flag:
                self.snaps_path = os.path.join(current_path, 'results_predators_prey')  # The resource folder path
                self.snaps_path = os.path.join(self.snaps_path, 'snaps')  # The resource folder path

        # self.cells = []
        # self.agents_positions_idx = []

        # self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]

        positions_idx = []

        if self.game_mode == 0:
            # first enter the positions for the agents (predators) and the single prey. If the grid is n*n,
            # then the positions are
            #  0                1             2     ...     n-1
            #  n              n+1           n+2     ...    2n-1
            # 2n             2n+1          2n+2     ...    3n-1
            #  .                .             .       .       .
            #  .                .             .       .       .
            #  .                .             .       .       .
            # (n-1)*n   (n-1)*n+1     (n-1)*n+2     ...   n*n+1
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0, 6, and 23 are the positions of the agents 24 is the position
            # of the prey
            positions_idx = []

        if self.game_mode == 1:
            positions_idx = np.random.choice(len(cells), size=self.num_predators + self.num_preys, replace=False)

        return [cells, positions_idx]

    def reset(self, pA=None, pB=None, ballOwner=None):  # initialize the world
        self.terminal = False
        if pA is not None:
            self.initPositions[0] = pA

        if pB is not None:
            self.initPositions[1] = pB

        if ballOwner is None:
            ballOwner = self.choosePlayer()

        self.positions = self.initPositions.copy()
        self.ballOwner = ballOwner
        return self.boardToState()

    def step(self, actions):
        actionA = actions[0]
        actionB = actions[1]
        if np.random.rand() < self.drawProbability:
            return self.boardToState(), self.resultToReward(-2, actionA, actionB), True, -2
        first = self.choosePlayer()
        actions = [actionA, actionB]
        m1 = self.move(first, actions[first])
        return_val = 0
        if (m1 >= 0):
            return_val = m1
        else:
            return_val = self.move(1-first, actions[1-first])
        if return_val!=-1:
            self.terminal = True
        return self.boardToState(), self.resultToReward(return_val,actionA, actionB), self.terminal, return_val
        # return self.move(1 - first, actions[1 - first])

    def move(self, player, action):
        opponent = 1 - player
        newPosition = self.positions[player] + self.actionToMove(action)

        # If it's opponent position
        if (newPosition == self.positions[opponent]).all():
            self.ballOwner = opponent
        # If it's the goal
        elif self.ballOwner is player and self.isInGoal(*newPosition) >= 0:
            return 1 - self.isInGoal(*newPosition)
        # If it's in board
        elif self.isInBoard(*newPosition):
            self.positions[player] = newPosition
        return -1
    
    def boardToState(self):
        # game = self.game
        xA, yA = self.positions[0]
        xB, yB = self.positions[1]
        sA = yA * self.w + xA
        sB = yB * self.w + xB
        sB -= 1 if sB > sA else 0
        state = (sA * (self.w * self.h - 1) + sB) + (self.w * self.h) * (self.w * self.h - 1) * self.ballOwner
        return [state]
    
    def resultToReward(self, result, actionA=None, actionB=None):
        if result >= 0:
            reward = (result*(-2) + 1)
        else:
            reward = 0
        return reward
    
    def actionToMove(self, action):
        switcher = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [0, 0],
        }
        return switcher.get(action)

    def isInGoal(self, x, y):
        g1, g2 = self.goalPositions
        if (g1 <= y <= g2):
            if x == -1:
                return 1
            elif x == self.w:
                return 0
        return -1

    def isInBoard(self, x, y):
        return (0 <= x < self.w and 0 <= y < self.h)
    
    def choosePlayer(self):
        return np.random.randint(0, 2)


    def action_space(self):
        return len(self.A)


    def draw(self, positions=None, ballOwner=None):
        positions = self.positions if positions is None else np.array(positions)
        ballOwner = self.ballOwner if ballOwner is None else ballOwner

        board = ''
        for y in range(self.h)[::-1]:
            for x in range(self.w):
                if ([x, y] == positions[0]).all():
                    board += 'A' if ballOwner == 0 else 'a'
                elif ([x, y] == positions[1]).all():
                    board += 'B' if ballOwner == 1 else 'b'
                else:
                    board += '-'
            board += '\n'

        print(board)

    def render(self):

        pygame.time.wait(500)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(BLACK)
        text = self.my_font.render("Step: {0}".format(self.step_num), 1, WHITE)
        self.screen.blit(text, (5, 15))

        # for row in range(self.grid_size):
        #     for column in range(self.grid_size):
        #         pos = (row, column)
        #         if pos in self.predators_positions and pos in self.preys_positions:
        #             color = ORANGE
        #         elif pos in self.predators_positions:
        #             color = BLUE
        #         elif pos in self.preys_positions:
        #             color = RED
        #         else:
        #             color = WHITE
        #         pygame.draw.rect(self.screen, color,
        #                          [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
        #                           HEIGHT])

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row, column)
                if pos in self.predators_positions and pos in self.preys_positions:
                    self.screen.blit(self.img_predator_prey,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                elif pos in self.predators_positions:
                    self.screen.blit(self.img_predator,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                elif pos in self.preys_positions:
                    self.screen.blit(self.img_prey,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                else:
                    color = WHITE
                    pygame.draw.rect(self.screen, color,
                                 [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                  HEIGHT])

        if self.recorder_flag:
            file_name = "%04d.png" % self.step_num
            pygame.image.save(self.screen, os.path.join(self.snaps_path, file_name))

        if not self.terminal:
            self.step_num += 1

    def gui_setup(self):

        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        board_size_x = (WIDTH + MARGIN) * self.grid_size
        board_size_y = (HEIGHT + MARGIN) * self.grid_size

        window_size_x = int(board_size_x*1.01)
        window_size_y = int(board_size_y * 1.2)

        window_size = [window_size_x, window_size_y]
        screen = pygame.display.set_mode(window_size, 0, 32)

        # Set title of screen
        pygame.display.set_caption("Predators-and-Prey Game")

        myfont = pygame.font.SysFont("monospace", 30)

        return [screen, myfont]
