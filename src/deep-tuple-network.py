'''
1. build a network
2. generate sample
3. backprogation
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random
from ExpectedMax import transpose, move_action
from Game import Game
import h5py

table = {2**i: i for i in range(1, 16)}
table[0] = 0


class DQN(object):
    def __init__(self):
        '''build a nn'''
        self.num_inputs = 8
        self.num_hiddens1 = 30
        self.num_hiddens2 = 30
        self.num_output = 1
        self.lr = 0.01
        self.batch_x = []
        self.batch_y = []
        self.build()

    def build(self):
        model = Sequential()
        model.add(Dense(self.num_hiddens1, init='lecun_uniform',
                        input_shape=(self.num_inputs, 4, )))
        model.add(Activation('relu'))

        model.add(Dense(self.num_hiddens2, init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(self.num_hiddens2, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(self.num_output, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model

    def update(self, grid, v, best_mat, input_x):
        r0 = grid[0]
        r1 = grid[1]
        r2 = grid[2]
        r3 = grid[3]
        grid = transpose(grid)
        r4 = grid[0]
        r5 = grid[1]
        r6 = grid[2]
        r7 = grid[3]
        x = np.array([r0, r1, r2, r3, r4, r5, r6, r7])
        old = sum(self.model.predict(
            np.array([x]), batch_size=1)[0]) / 8

        new = (v - old)
        self.batch_x.append(x)
        self.batch_y.append([new] * 8)
        K = 1000
        if len(self.batch_x) >= K:
            indices = random.sample(range(K), int(K * 0.4))
            self.model.fit(np.array([self.batch_x[i] for i in (indices)]), np.array(
                [self.batch_y[i] for i in (indices)]), batch_size=int(K * 0.4), epochs=1, verbose=0)

            self.batch_x.remove(self.batch_x[0])
            self.batch_y.remove(self.batch_y[0])

    def make_desicion(self, grid):
        '''
        put 4 actions in the NN and tried to predict Q value
        '''
        g = grid
        best_mat = None
        best_move = -1
        best_v = -100
        for m in range(4):
            mat, done, s = move_action(g, m)
            tmat = transpose(mat)
            input_x = []
            if done == True:
                r0 = (mat[0])
                r1 = (mat[1])
                r2 = (mat[2])
                r3 = (mat[3])
                r4 = (tmat[0])
                r5 = (tmat[1])
                r6 = (tmat[2])
                r7 = (tmat[3])
                input_x = np.array(
                    [r0, r1, r2, r3, r4, r5, r6, r7])
                v = sum(self.model.predict(
                    np.array([input_x]), batch_size=1)[0]) / 8

                if best_v < 2 * s + v:
                    best_move = m
                    best_v = s * 2 + v
                    best_mat = mat
                    input_x_best = input_x
        if best_move < 0:
            best_v = -100000
        self.update(grid, best_v, best_mat, input_x_best)

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
    print_result(game.grid)

    return max([max(i) for i in game.grid])


def get_score_new(grid):
    tgrid = transpose(grid)
    a = sum([sum(grid[i]) * (2 << i) for i in range(len(grid))])
    a += sum([sum(tgrid[i]) * (2 << i) for i in range(len(tgrid))])
    return a


def make_input(grid):
    global table
    r = np.zeros(shape=(16, 4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            v = grid[i, j]
            r[table[v], i, j] = 1
    return r


def print_result(grid):
    for i in grid:
        print(i)


dqn = DQN()
for i in range(10000):
    print(str(i) + ":" + str(play_game()))
dqn.model.save('new.h5')
