
from chromectrl import ChromeDebuggerControl
from gamectrl import Fast2048Control
import Expectimax
from Game import Game
from deep_reinforcement_network import DQN
import time
import os

E = Expectimax.ExpectMax()
D = DQN()
Expectimax.build_table()


def find_best_move_e(board,type):
    G = Game(4, grid=board)
    if type=="expectimax":
        return E.get_move(G)
    else:
        return D.make_desicion(board)


def find_best_move_d(board):
    return D.make_desicion(board)


def play_game(gamectrl,name):
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
        move = find_best_move_e(board,name)

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
    strategy = input("Which type of strategy you want to test? Expectimax input 1, Deep Q learning input 2\n")
    if strategy == "1":
        play_game(gamectrl,"expectimax")
    else:
        play_game(gamectrl,"DQN")

if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
