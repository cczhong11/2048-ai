# cython: infer_types=True,  infer_types.verbose=False, boundscheck=False, initializedcheck=False
# cython: annotation_typing=True, cdivision=True


from __future__ import print_function

import os
import sys
import copy
import random
import functools
from Game import Game

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
# Python 2/3 compatibility.
if sys.version_info[0] == 2:
    range = xrange
    input = raw_input


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
        max_depth = 1
        grid_copy = copy.deepcopy(game.grid)
        for move in range(4):
            # print_result(grid_copy)
            (grid, tdone) = move_gride(grid_copy, move)
            # print_result(grid)
            # print("------------\n")
            if tdone == True:
                score[move] = self.get_expected(grid, max_depth)
            else:
                score[move] = 0
            # print(score[move])
            done = tdone or done
        # print(score)
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
        empties = get_empty_cells(grid)
        num_empty = len(empties)
        sumexpected = 0
        # print(num_empty)
        if depth == 0:
            # print(".")
            return self.get_score(grid, num_empty)
        if num_empty == 0:
            sumexpected = self.try_move(grid, depth - 1)
        else:
            # for i in range(self.n):
            #    for j in range(self.n):
            newgrid2 = grid
            newgrid4 = grid
            tmp = []
            for times in range(num_empty):
                k = empties[times]
                # print(empty_set[0])
                j, i = k
                if grid[j][i] == 0:
                    newgrid2[j][i] = 2
                    sumexpected += self.try_move(newgrid2, depth - 1) * 0.9
                    newgrid4[j][i] = 4
                    sumexpected += self.try_move(newgrid4, depth - 1) * 0.1
                tmp.append(k)
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


def _getch_windows(prompt):
    """
    Windows specific version of getch.  Special keys like arrows actually post
    two key events.  If you want to use these keys you can create a dictionary
    and return the result of looking up the appropriate second key within the
    if block.
    """
    print(prompt, end="")
    key = msvcrt.getch()
    if ord(key) == 224:
        key = msvcrt.getch()
        return key
    print(key.decode())
    return key.decode()


def _getch_linux(prompt):
    """Linux specific version of getch."""
    print(prompt, end="")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    char = None
    try:
        char = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
    print(char)
    return char


# Set version of getch to use based on operating system.
if sys.platform[:3] == 'win':
    import msvcrt
    getch = _getch_windows
else:
    import termios
    getch = _getch_linux


def push_row(row, left=True):
    """Push all tiles in one row; like tiles will be merged together."""
    row = row[:] if left else row[::-1]
    new_row = [item for item in row if item]
    for i in range(len(new_row) - 1):
        if new_row[i] and new_row[i] == new_row[i + 1]:
            new_row[i], new_row[i + 1:] = new_row[i] * \
                2, new_row[i + 2:] + [0]
    new_row += [0] * (len(row) - len(new_row))
    return new_row if left else new_row[::-1]


def get_column(grid, column_index):
    """Return the column from the grid at column_index  as a list."""
    return [row[column_index] for row in grid]


def set_column(grid, column_index, new):
    """
    Replace the values in the grid at column_index with the values in new.
    The grid is changed inplace.
    """
    for i, row in enumerate(grid):
        row[column_index] = new[i]


def push_all_rows(grid, left=True):
    """
    Perform a horizontal shift on all rows.
    Pass left=True for left and left=False for right.
    The grid will be changed inplace.
    """
    for i, row in enumerate(grid):
        grid[i] = push_row(row, left)


def push_all_columns(grid, up=True):
    """
    Perform a vertical shift on all columns.
    Pass up=True for up and up=False for down.
    The grid will be changed inplace.
    """
    for i, val in enumerate(grid[0]):
        column = get_column(grid, i)
        new = push_row(column, up)
        set_column(grid, i, new)


def get_empty_cells(grid):
    """Return a list of coordinate pairs corresponding to empty cells."""
    empty = []
    for j, row in enumerate(grid):
        for i, val in enumerate(row):
            if not val:
                empty.append((j, i))
    return empty


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    if get_empty_cells(grid):
        return True
    for row in grid:
        if any(row[i] == row[i + 1] for i in range(len(row) - 1)):
            return True
    for i, val in enumerate(grid[0]):
        column = get_column(grid, i)
        if any(column[i] == column[i + 1] for i in range(len(column) - 1)):
            return True
    return False


def get_start_grid(cols=4, rows=4):
    """Create the start grid and seed it with two numbers."""
    grid = [[""] * cols for i in range(rows)]
    for i in range(2):
        empties = get_empty_cells(grid)
        y, x = random.choice(empties)
        grid[y][x] = 2 if random.random() < 0.9 else 4
    return grid


def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = get_empty_cells(grid)
    y, x = random.choice(empties)
    grid[y][x] = 2 if random.uniform(0, 1) < 0.9 else 4
    return any_possible_moves(grid)


def print_grid(grid):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------" * len(grid[0]) + "+"
    print(wall)
    for row in grid:
        meat = "|".join("{:^6}".format(val) for val in row)
        print("|{}|".format(meat))
        print(wall)


def move_gride(mat, move):
    done = True
    mat_copy = copy.deepcopy(mat)
    if move == 0:
        push_all_columns(mat_copy, up=True)
    if move == 1:
        push_all_columns(mat_copy, up=False)
    if move == 2:
        push_all_rows(mat_copy, left=True)
    if move == 3:
        push_all_rows(mat_copy, left=False)
    if mat == mat_copy:
        done = False
    return (mat_copy, done)


def print_result(grid):
    for i in grid:
        print(i)


def main():
    game = Game(4)
    game.add_two()
    game.add_two()
    E = ExpectMax()
    # print_result(game.grid)
    print(game.game_state())
    while game.game_state() == 'not over':
        done = E.get_move(game)
        # print(done)
        # print_result(game.grid)
        if done < 0:
            print(str(done) + "end of game")
            print_result(game.grid)
            input()
            break
        if done == 0:
            push_all_columns(game.grid, up=True)
        if done == 1:
            push_all_columns(game.grid, up=False)
        if done == 2:
            push_all_rows(game.grid, left=True)
        if done == 3:
            push_all_rows(game.grid, left=False)
        #print(str(done) + "\n------------\n")
        # print_result(game.grid)
        #print(max([max(i) for i in game.grid]))
        game.add_two()

        # E.print_result(game.grid)
        # print("----\n")
    # print(game.game_state())
    # print_result(game.grid)
    # input()
    return max([max(i) for i in game.grid])


# def main():
#    """
#    Get user input.
#    Update game state.
#    Display updates to user.
#    """
#    functions = {"a": functools.partial(push_all_rows, left=True),
#                 "d": functools.partial(push_all_rows, left=False),
#                 "w": functools.partial(push_all_columns, up=True),
#                 "s": functools.partial(push_all_columns, up=False)}
#    grid = get_start_grid(*map(int, sys.argv[1:]))
#    print_grid(grid)
#    while True:
#        grid_copy = copy.deepcopy(grid)
#        get_input = getch("Enter direction (w/a/s/d): ")
#        if get_input in functions:
#            functions[get_input](grid)
#        elif get_input == "q":
#            break
#        else:
#            print("\nInvalid choice.")
#            continue
#        if grid != grid_copy:
#            if not prepare_next_turn(grid):
#                print_grid(grid)
#                print("You Lose!")
#                break
#        print_grid(grid)
#    print("Thanks for playing.")
#

if __name__ == "__main__":
    m = {}
    for i in range(1000):
        print(i)
        maxscore = main()
        if maxscore in m:
            m[maxscore] += 1
        else:
            m[maxscore] = 1
        print(m)
    print(m)
