'''this file is for ExpectMax algo implement'''
from Game import Game
import random
import logging
logging.basicConfig(filename="lograte.txt", filemode="w", level=logging.INFO)
EMPTY_SCORE = 10
MONO_SCORE = 2
MERGE_SCORE = 2
MAX_IN_BOARD = 2
EMPTY_l = []
MERGE_l = []
Mono_l = []
MAX_in_b = []


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
        for move in range(4):
            (grid, tdone) = move_gride(game.grid, move)
            if tdone == True:
                score[move] = self.get_expected(grid, max_depth)
            else:
                score[move] = 0
            done = tdone or done

        if done == False or max(score) == 0:
            return -1
        else:
            # print(max(score))
            return score.index(max(score))

    'use in E to get max expected score'

    def try_move(self, grid, max_depth):
        ''' get move for now situation
        return max Expected score
        '''
        done = False
        score = [0 for _ in range(4)]
        grids = []
        for move in range(4):
            (grid, tdone) = move_gride(grid, move)
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
            # for i in range(self.n):
            #    for j in range(self.n):
            newgrid2 = grid
            newgrid4 = grid
            tmp = []
            for times in range(depth + 2):
                k = random.randint(0, num_empty - 1)
                if k in tmp:
                    continue
                # print(empty_set[0])
                i = empty_set[k][0]
                j = empty_set[k][1]
                if grid[i][j] == 0:
                    newgrid2[i][j] = 2
                    sumexpected += self.try_move(newgrid2, depth - 1) * 0.9
                    newgrid4[i][j] = 4
                    sumexpected += self.try_move(newgrid4, depth - 1) * 0.1
                tmp.append(k)
            sumexpected = sumexpected / num_empty
        return sumexpected

    # def get_score(self, grid, num_empty):
    #    return num_empty
    #'function to get scores '

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

        emscore = num_empty ** EMPTY_SCORE
        maxnum = max([max(i) for i in grid])
        maxbscore = 0
        for i in [0, self.n - 1]:
            if grid[i].index(max(grid[i])) in [0, self.n - 1] and max(grid[i]) > 0:
                maxbscore += max(grid[i])**MAX_IN_BOARD
        mescore = 0
        tmscore = 0
        for i in range(self.n - 1):
            mscore1 = 0
            mscore2 = 0
            countml = 0
            countmr = 0
            for j in range(self.n - 1):
                if grid[i][j] == grid[i][j + 1] and grid[i][j] > 0:
                    mescore += grid[i][j]**MERGE_SCORE
                    if grid[i][j] == maxnum:
                        mescore += grid[i][j]**(MERGE_SCORE)
                if grid[i][j] == grid[i + 1][j] and grid[i][j] > 0:
                    mescore += grid[i][j]**MERGE_SCORE
                    if grid[i][j] == maxnum:
                        mescore += grid[i][j]**MERGE_SCORE
                if grid[i][j] > grid[i][j + 1]:
                    mscore1 += (grid[i][j] - grid[i][j + 1])**MONO_SCORE
                    countml += 1
                if grid[i][j] < grid[i][j + 1]:
                    mscore2 += (grid[i][j + 1] - grid[i][j])**MONO_SCORE
                    countmr += 1
            if countml == 4 or countml == 3:
                tmscore += mscore1
            if countmr == 4 or countmr == 3:
                tmscore += mscore2
        # score += max([max(i) for i in grid])
        score = emscore + tmscore + mescore + maxbscore
        EMPTY_l.append(emscore / score)
        Mono_l.append(tmscore / score)
        MERGE_l.append(mescore / score)
        MAX_in_b.append(maxbscore / score)
        # logging.info(str(emscore/score)+"\t"+str(tmscore/score)+"\t"+str(mescore/score)+"\t"+str(maxbscore/score)+"\t\n")
        return score

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


def move_gride(mat, move):
    if move == 0:
        mat = transpose(mat)
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = transpose(mat)
    elif move == 1:
        mat = reverse(transpose(mat))
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = transpose(reverse(mat))
    elif move == 2:
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
    elif move == 3:
        mat = reverse(mat)
        mat, done = cover_up(mat)
        temp = merge(mat)
        mat = temp[0]
        done = done or temp[1]
        mat = cover_up(mat)[0]
        mat = reverse(mat)
    return (mat, done)


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
        # E.print_result(game.grid)
        # print("----\n")
        print(max([max(i) for i in game.grid]))
    return max([max(i) for i in game.grid])


if __name__ == '__main__':
    m = {}
    for i in range(100):
        print(i)
        maxscore = main()
        if maxscore in m:
            m[maxscore] += 1
        else:
            m[maxscore] = 1
        print(m)
    print(m)
    print(sum(EMPTY_l) / len(EMPTY_l))
    print(sum(Mono_l) / len(Mono_l))
    print(sum(MERGE_l) / len(MERGE_l))
    print(sum(MAX_in_b) / len(MAX_in_b))
