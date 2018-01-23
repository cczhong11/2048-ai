# AI for 2048

2048 is a very popular online game. It is very easy but hard to achieve its goal. Therefore we decided to develop an AI agent to solve the game. We explored two strategies in our project, one is ExpectiMax and the other is Deep Reinforcement Learning. In ExpectiMax strategy, we tried 4 different heuristic functions and combined them to improve the performance of this method. In deep reinforcement learning, we used sum of grid as reward and trained two hidden layers neural network. For ExpectiMax method, we could achieve **98%** in 2048 with setting depth limit to 3. But we didn't achieve a good result in deep reinforcement learning method, the max tile we achieved is 512.

# how to use

all the file should use python 2.7 to run.

if you want to use chrome ctrl

- for mac user enter `/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome http://gabrielecirulli.github.io/2048/ --remote-debugging-port=9222 ` in terminal
- run `python2 2048.py`


# Game infrusture


The game logic is used code from [2048-python](https://github.com/yangshun/2048-python).

The game contrl part code are used from [2048-ai](https://github.com/nneonneo/2048-ai)
# ExpectiMax

The class is in `src\Expectimax\ExpectedMax.py`. 

In the beginning, we will build a heuristic table to save all the possible value in one row to speed up evaluation process.

In each state, it will call `get_move` to try different actions, and afterwards, it will call `get_expected` to put 2 or 4 in empty tile. Then depth +1 , it will call `try_move` in the next step.

# Deep reinforcement learning

The main class is in `deep-reinforcement-learning*.py`.

The first version in just a draft, the second one use CNN as an architecture, and this method could achieve 1024, but its result actually not very depend on the predict result. The third version I implement a strategy that move action totally reply on the output of neural network. The result is not satsified, the highest score I achieve is only 512.


