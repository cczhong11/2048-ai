'''this file is for ExpectMax algo implement'''
import math
from Game import Game
import random
import copy


heurs_table = [0 for _ in range(65536)]
dict_2 = {1 << i: i for i in range(17)}
dict_2[0] = 0


def build_table():
    global heurs_table
    for row in range(65536):
        empty = 0
        merge = 0
        mono = 0
        num = [0 for _ in range(4)]
        num[0] = row >> 12 & 15
        num[1] = row >> 8 & 15
        num[2] = row >> 4 & 15
        num[3] = row & 15
        for i in range(4):
            if num[i] == 0:
                empty += 1
            if i > 0 and num[i] == num[i - 1]:
                merge += num[i]
        heurs_table[row] = empty + 10 * merge


class ExpectMax(object):
    ''' choose max exp utilize value to move  '''

    def __init__(self):
        self.transcore = {}
        self.n = 4

    def get_move(self, game):
        ''' get move for now situation
            return move
        '''
        done = False
        score = [0 for _ in range(4)]
        max_depth = 2
        mat_copy = copy.deepcopy(game.grid)
        for move in range(4):
            (grid, tdone) = move_gride(mat_copy, move)
            if tdone == True:
                score[move] = self.get_expected(grid, max_depth)
            else:
                score[move] = 0
            done = tdone or done

        if done == False or max(score) == 0:
            return -1
        else:
            return score.index(max(score))

    'use in E to get max expected score'

    def try_move(self, grid, max_depth):
        ''' get move for now situation
        return max Expected score
        '''
        done = False
        score = [0 for _ in range(4)]
        grids = []
        grid_copy = copy.deepcopy(grid)
        for move in range(4):
            (grid, tdone) = move_gride(grid_copy, move)
            if tdone == True:
                score[move] = self.get_expected(grid, max_depth)
            else:
                score[move] = 0
            done = tdone or done
        if done == False:
            return 0
        else:
            return max(score)

    'function to get E score for each move'

    def get_expected(self, grid, depth):
        (num_empty, empty_set) = get_empty(grid)
        sumexpected = 0
        if depth == 0:
            return self.get_score(grid, num_empty)
        if num_empty == 0:
            sumexpected = self.try_move(grid, depth - 1)
        else:
           
            newgrid2 = grid
            newgrid4 = grid
            tmp = []
            for times in range(num_empty):

                i = empty_set[times][0]
                j = empty_set[times][1]
                if grid[i][j] == 0:
                    newgrid2[i][j] = 2
                    sumexpected += self.try_move(newgrid2, depth - 1) * 0.9
                    newgrid4[i][j] = 4
                    sumexpected += self.try_move(newgrid4, depth - 1) * 0.1

            sumexpected = sumexpected / num_empty
        return sumexpected

    

    def get_score2(self, grid, num_empty):
        score = 0
        n = 1
        for i in range(self.n):
            if i % 2 == 0:
                for j in range(self.n):
                    score += grid[i][j] * 4**n
            if i % 2 == 1:
                for j in range(self.n - 1, -1, -1):
                    score += grid[i][j] * 4**n
            n += 1
        return score

    def get_score(self, grid, num_empty):
        global heurs_table
        grid2 = transpose(grid)
        sum_socre = 0
        for i in grid:

            row = (dict_2[i[3]] << 12) + (dict_2[i[2]] << 8) + \
                (dict_2[i[1]] << 4) + dict_2[i[0]]

            sum_socre += heurs_table[row]
        for i in grid2:
            row = (dict_2[i[3]] << 12) + (dict_2[i[2]] << 8) + \
                (dict_2[i[1]] << 4) + dict_2[i[0]]
            sum_socre += heurs_table[row]
        return sum_socre
        

    def print_result(self, grid):
        for i in grid:
            print(i)


def get_empty(grid):
    '''get empty number of tiles'''
    l = []
    empty_size = 0
    for i in range(4):
        for j in range(4):
            if grid[i][j] == 0:
                l.append((i, j))
                empty_size += 1
    return (empty_size, l)

# def get_index(grid):
#    return sum([])


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0]) - j - 1])
    return new


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def cover_up(mat):

    new = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)


def merge(mat):
    done = False
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j + 1] = 0
                done = True
    return (mat, done)


def merge_score(mat):
    done = False
    score = 0
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                score += mat[i][j]
                mat[i][j + 1] = 0
                done = True
    return (mat, done, score)


def move_gride(matt, move):
    mat_copy = copy.deepcopy(matt)
    if move == 0:
        mat = transpose(mat_copy)
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = transpose(mat)
    elif move == 1:
        mat = reverse(transpose(mat_copy))
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = transpose(reverse(mat))
    elif move == 2:
        mat, done = cover_up(mat_copy)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
    elif move == 3:
        mat = reverse(mat_copy)
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = reverse(mat)

    return (mat, done)


def get_score_s(grid):
    score = 0
    n = 0
    Mono_l = 0
    Mono_r = 0
    mono = 0
    for i in range(4):

        if i % 2 == 0:
            for j in range(4):
                score += grid[i][j] * 4**n
                n += 1
        if i % 2 == 1:
            for j in range(4 - 1, -1, -1):
                score += grid[i][j] * 4**n
                n += 1

        '''
        Mono_l = 0
        Mono_r = 0
        for j in range(1, 4):
            if grid[i][j] - grid[i][j - 1] > 0:
                Mono_l += grid[i][j] - grid[i][j - 1]
            else:
                Mono_r += grid[i][j - 1] - grid[i][j]
        mono += - min(Mono_l, Mono_r)'''
    return score


def move_action(matt, move):
    mat_copy = copy.deepcopy(matt)

    if move == 0:
        mat = transpose(mat_copy)
        mat, done = cover_up(mat)
        mat, sdone, score = merge_score(mat)
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = transpose(mat)
    elif move == 1:
        mat = reverse(transpose(mat_copy))
        mat, done = cover_up(mat)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = transpose(reverse(mat))
    elif move == 2:
        mat, done = cover_up(mat_copy)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
    elif move == 3:
        mat = reverse(mat_copy)
        mat, done = cover_up(mat)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = reverse(mat)
    score = get_score_s(mat)
    if done == False:
        score = -10000
    return (mat, done, score)


def move_action2(matt, move):
    mat_copy = copy.deepcopy(matt)

    if move == 0:
        mat = transpose(mat_copy)
        mat, done = cover_up(mat)
        mat, sdone, score = merge_score(mat)
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = transpose(mat)
    elif move == 1:
        mat = reverse(transpose(mat_copy))
        mat, done = cover_up(mat)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = transpose(reverse(mat))
    elif move == 2:
        mat, done = cover_up(mat_copy)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
    elif move == 3:
        mat = reverse(mat_copy)
        mat, done = cover_up(mat)
        temp, sdone, score = merge_score(mat)
        mat = temp
        done = done or sdone
        mat = cover_up(mat)[0]
        mat = reverse(mat)

    score += get_score_s(mat)
    #score += mat[0][0] * 12 + mat[0][1] * 8 + mat[0][2] * 4 + mat[0][3]

    return (mat, done, score)


def main():
    game = Game(4)
    game.add_two()
    game.add_two()
    E = ExpectMax()
    print(game.game_state())
    while game.game_state() == 'not over':
        done = E.get_move(game)
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
        E.print_result(game.grid)
    return max([max(i) for i in game.grid])



if __name__ == '__main__':
    m = {}
    build_table()
    for i in range(10):
        print(i)
        maxscore =  main()
        if maxscore in m:
            m[maxscore] += 1
        else:
            m[maxscore] = 1
        print(m)
    print(m)

