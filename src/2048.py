
from chromectrl import ChromeDebuggerControl
from gamectrl import Fast2048Control
import Expectimax
from Game import Game
from deep_reinforcement_network3 import DQN
import time
import os

E = Expectimax.ExpectMax()
D = DQN()
Expectimax.build_table()


def find_best_move_e(board):
    G = Game(4, grid=board)
    return E.get_move(G)


def find_best_move_d(board):
    return D.make_desicion(board)


def play_game(gamectrl):
    moveno = 0
    start = time.time()
    while 1:
        state = gamectrl.get_status()
        if state == 'ended':
            break
        elif state == 'won':
            time.sleep(0.75)
            gamectrl.continue_game()

        moveno += 1
        board = gamectrl.get_board()
        move = find_best_move_e(board)

        if move < 0:
            break
        #print("%010.6f: Score %d, Move %d: %s" % (time.time() - start, gamectrl.get_score(), moveno, movename(move)))
        gamectrl.execute_move(move)

    score = gamectrl.get_score()
    board = gamectrl.get_board()
    #maxval = max(max(row) for row in to_val(board))
    #print("Game over. Final score %d; highest tile %d." % (score, maxval))


def main(argv):

    ctrl = ChromeDebuggerControl(9222)

    gamectrl = Fast2048Control(ctrl)
    if gamectrl.get_status() == 'ended':
        gamectrl.restart_game()

    play_game(gamectrl)


if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
