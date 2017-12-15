'''
1. build a network
2. generate sample
3. backprogation
'''

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
import copy
import Expectimax

from Game import Game
import h5py

table = {2**i: i for i in range(1, 16)}
table[0] = 0
input_shape = (4, 4, 16)


class DQN(object):
    def __init__(self):
        '''build a nn'''
        self.num_inputs = 16
        self.num_hiddens1 = 512
        self.num_hiddens2 = 4096
        self.num_output = 1
        self.lr = 0.01
        self.batch_x = []
        self.batch_y = []
        # self.build()
        self.model = load_model('bestnnresult/new_c_4.h5')

    def build(self):
        model = Sequential()
        model.add(Conv2D(512, kernel_size=(2, 2),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(self.num_hiddens1, (2, 2), activation='relu'))

        model.add(Flatten())
        model.add(Dense(self.num_output, init='lecun_uniform'))

        model.add(Activation('linear'))
        print(model.output_shape)
        rms = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model

    def update(self, grid, v, best_mat, input_x):
        x = input_x
        g0 = make_input(grid)
        g1 = g0[:, ::-1, :]
        g2 = g0[:, :, ::-1]
        g3 = g2[:, ::-1, :]
        g4 = g0.swapaxes(0, 1)
        g5 = g4[:, ::-1, :]
        g6 = g4[:, :, ::-1]
        g7 = g5[:, ::-1, :]
        old = self.model.predict(np.array([g0]), batch_size=1)[0][0]
        new = (v - old)
        self.batch_x.append(np.array(g0))
        self.batch_x.append(np.array(g1))
        self.batch_x.append(np.array(g2))
        self.batch_x.append(np.array(g3))
        self.batch_x.append(np.array(g4))
        self.batch_x.append(np.array(g5))
        self.batch_x.append(np.array(g6))
        self.batch_x.append(np.array(g7))
        [self.batch_y.append(new) for i in range(8)]

        # x = np.array([g0, g1, g2, g3, g4, g5, g6, g7]).reshape(8, 4, 4, 16)
        # self.model.fit(x, np.array(
        #    [new] * 8), batch_size=8, verbose=0)

        K = 1000
        if len(self.batch_x) >= K:
            indices = random.sample(range(K), int(K * 0.6))
            self.model.fit(np.array([self.batch_x[i] for i in (indices)]), np.array(
                [self.batch_y[i] for i in (indices)]), batch_size=int(K * 0.6), epochs=1, verbose=0)

            self.batch_x.remove(self.batch_x[0])
            self.batch_y.remove(self.batch_y[0])

    def make_desicion(self, grid):
        '''
        put 4 actions in the NN and tried to predict Q value
        '''
        g = copy.deepcopy(grid)
        best_mat = None
        best_move = -1
        best_v = None
        input_x_best = None
        for m in range(4):
            mat, done, s = Expectimax.move_action(g, m)

            input_x = []
            if done == True:

                input_x = make_input(grid)
                v = self.model.predict(
                    np.array([input_x]), batch_size=1)[0][0]

                if best_v is None or best_v <= 2 * s + v:
                    if best_v == 2 * s + v:
                        if random.uniform(0, 1) < 0.5:
                            continue
                    best_move = m
                    best_v = s * 2 + v
                    best_mat = mat
                    input_x_best = input_x
        if best_move < 0:
            best_v = -100000
        # self.update(grid, best_v, best_mat, input_x_best)

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
        # print(max([max(i) for i in game.grid]))
        # print_result(game.grid)
        # print("----\n")

    # if max([max(i) for i in game.grid]) < 33:
    print_result(game.grid)

    return max([max(i) for i in game.grid])


def get_score_new(grid):
    tgrid = Expectimax.transpose(grid)
    a = sum([sum(grid[i]) * (2 << i) for i in range(len(grid))])
    a += sum([sum(tgrid[i]) * (2 << i) for i in range(len(tgrid))])
    return a


def make_input(grid):
    global table
    r = np.zeros(shape=(4, 4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            v = grid[i][j]
            r[i, j, table[v]] = 1
    return r


def print_result(grid):
    for i in grid:
        print(i)


dqn = DQN()
mm = {}
for i in range(10000):
    smax = play_game()
    print(str(i) + ":" + str(smax))
    if smax not in mm:
        mm[smax] = 1
    else:
        mm[smax] += 1
    print(mm)
    # if smax > 511:
    #    dqn.model.save('t.h5')
