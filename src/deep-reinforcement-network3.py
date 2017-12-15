'''
1. build a network
2. generate sample
3. backprogation
'''

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adagrad, Adamax
import time
import numpy as np
import random
import Expectimax
import matplotlib.pyplot as plt

from Game import Game
import h5py
import copy

table = {2**i: i for i in range(1, 16)}
table[0] = 0


class DQN(object):
    def __init__(self):
        '''build a nn'''
        self.num_inputs = 16
        self.num_hiddens1 = 256
        self.num_hiddens2 = 64
        self.num_output = 4
        self.lr = 0.00005
        self.batch_x = []
        self.batch_y = []
        # self.model = load_model('simpleQ4max1512568239.76.h5')
        self.build()
        self.loss = 0

    def build(self):
        model = Sequential()
        model.add(Dense(self.num_hiddens1, init='lecun_uniform',
                        input_shape=(self.num_inputs,)))
        model.add(Activation('relu'))

        model.add(Dense(self.num_hiddens2, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(self.num_output, init='lecun_uniform'))
        model.add(Activation('linear'))

        # very goodrms = Adagrad(lr=self.lr)
        rms = Adagrad(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model

    def update(self, grid, g, chose_move, result):
        if chose_move > -1:
            score = np.zeros(4)
            for i in range(4):
                new_mat, done, score[i] = Expectimax.move_action2(grid, i)
                if i == chose_move:
                    gg = make_input(new_mat)
                    new_result = self.model.predict(
                        np.array([gg]), batch_size=1)[0]
            if score.ptp() != 0:
                score = (score - score.mean()) / score.ptp() * 100
            self.batch_x.append(g)
            result[chose_move] = score[chose_move] + \
                0.01 * np.max(new_result[0])
            self.batch_y.append(result)
        K = 1000
        if len(self.batch_x) >= K:
            indices = random.sample(range(K), int(K * 0.5))
            result = self.model.fit(np.array([self.batch_x[i] for i in (indices)]), np.array(
                [self.batch_y[i] for i in (indices)]), batch_size=int(K * 0.5), epochs=1, verbose=0)
            self.loss = result.history['loss'][0]
            self.batch_x.remove(self.batch_x[0])
            self.batch_y.remove(self.batch_y[0])

    def make_desicion(self, grid):
        '''
        put 4 actions in the NN and tried to predict Q value
        '''
        g = make_input(grid)

        result = self.model.predict(np.array([g]), batch_size=1)[0]
        chose_move = np.argmax(result)
        best_move = chose_move
        new_mat, done, score = Expectimax.move_action(grid, best_move)
        flag = 0
        if done == False:
            for i in range(4):
                new_mat, done, score = Expectimax.move_action(grid, i)
                if done == True:
                    flag = 1
                    best_move = i
            if flag == 0:
                best_move = -1
        origin_mat = copy.deepcopy(grid)
        self.update(origin_mat, g, best_move, result)

        return best_move


def play_game():
    game = Game(4)
    game.add_two()
    game.add_two()

    print(game.game_state())
    while game.game_state() == 'not over':
        done = dqn.make_desicion(game.grid)
        # print(done)
        if done < 0:

            print("end of game")
            break
        if done == 0:
            game.up()
        if done == 1:
            game.down()
        if done == 2:
            game.left()
        if done == 3:
            game.right()
        game.add_two()

        # print_result(game.grid)
        # print("----\n")

    # if max([max(i) for i in game.grid]) < 33:
    # print_result(game.grid)

    return max([max(i) for i in game.grid])


def get_score_new(grid):
    tgrid = transpose(grid)
    a = sum([sum(grid[i]) * (2 << i) for i in range(len(grid))])
    a += sum([sum(tgrid[i]) * (2 << i) for i in range(len(tgrid))])
    return a


def make_input(grid):
    global table
    r = np.zeros(shape=(16), dtype=int)
    k = 0
    for i in range(4):
        for j in range(4):
            v = grid[i][j]
            r[k] = table[v]
            k += 1
    return r


def print_result(grid):
    for i in grid:
        print(i)


dqn = DQN()
mm = {}
for i in range(10000):
    smax = play_game()
    if smax not in mm:
        mm[smax] = 1
    else:
        mm[smax] += 1
    print(str(i) + ":" + str(smax))
    print(str(mm))
    plt.scatter(i, dqn.loss)
    plt.pause(0.1)

    if smax > 511:
        dqn.model.save('simpleQmax512_' + str(i) + '_.h5')
    if smax > 1023:
        dqn.model.save('simplenn1024_' + str(i) + '_time.h5')
    if smax > 1024:
        dqn.model.save('simplenn2048_' + str(i) + '_time.h5')

dqn.model.save('new.h5')
