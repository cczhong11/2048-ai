'''this file is for ExpectMax algo implement'''
from Game import Game
import random

class ExpectMax(object):
    ''' choose max exp utilize value to move  '''
    def __init__(self):
        self.transcore = {}
        self.n = 4
    
    def get_move(self, game):
        ''' get move for now situation'''    
        done = False
        score = [0 for _ in range(4)]
        max_depth = 2
        for move in range(4):
            if move == 0:
                (grid, tdone) = up(game.grid)
                if tdone == True:
                    score[0] = self.get_expected(grid,max_depth)
                done = tdone or done
            if move == 1:
                (grid, tdone) = down(game.grid)
                if tdone == True:
                    score[1] = self.get_expected(grid,max_depth)
                done = tdone or done
            if move == 2:
                (grid, tdone) = left(game.grid)
                if tdone == True:
                    score[2] = self.get_expected(grid,max_depth)
                done = tdone or done
            if move == 3:
                (grid, tdone) = right(game.grid)
                if tdone == True:
                    score[3] = self.get_expected(grid,max_depth)
                done = tdone or done
        if done == False:
            return -1
        else:
            return score.index(max(score))
            
            
    def get_expected(self, grid, depth):
        num_empty = get_empty(grid)
        sumexpected = 0
        if num_empty == 0:
            sumexpected = self.get_score(grid,0)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    newgrid2 = grid
                    newgrid4 = grid
                    if grid[i][j] == 0:
                        newgrid2[i][j] = 2
                        sumexpected += self.get_score(newgrid2,num_empty-1)*0.5
                        newgrid4[i][j] = 4
                        sumexpected += self.get_score(newgrid4,num_empty-1)*0.5
            
            sumexpected = sumexpected/num_empty
        return sumexpected


    def get_score(self,grid,num_empty):
        score = num_empty * 10
        for i in [0,self.n-1]:
            if grid[i].index(max(grid[i])) in [0,self.n-1] and max(grid[i])>0:
                score += max(grid[i])
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if grid[i][j] == grid[i][j+1] and grid[i][j] > 0:
                    score += 5*grid[i][j]
                if grid[i][j] == grid[i+1][j] and grid[i][j] > 0:
                    score += 5*grid[i][j]
        score += 10*max([max(i) for i in grid])
        return score
    def print_result(self, grid):
        for i in grid:
            print(i)

def get_empty(grid):
    '''get empty number of tiles'''
    return sum([e.count(0) for e in grid])

#def get_index(grid):
#    return sum([])

def reverse(mat):
    new=[]
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

def transpose(mat):
    new=[]
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

def cover_up(mat):
    new=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    done=False
    for i in range(4):
        count=0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count]=mat[i][j]
                if j!=count:
                    done=True
                count+=1
    return (new,done)

def merge(mat):
    done=False
    for i in range(4):
         for j in range(3):
             if mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                 mat[i][j]*=2
                 mat[i][j+1]=0
                 done=True
    return (mat,done)


def up(mat):
    mat=transpose(mat)
    mat,done=cover_up(mat)
    temp=merge(mat)
    mat=temp[0]
    done=done or temp[1]
    mat=cover_up(mat)[0]
    mat=transpose(mat)
    return (mat,done)

def down(mat):
    mat=reverse(transpose(mat))
    mat,done=cover_up(mat)
    temp=merge(mat)
    mat=temp[0]
    done=done or temp[1]
    mat=cover_up(mat)[0]
    mat=transpose(reverse(mat))
    return (mat,done)

def left(mat):
    mat,done=cover_up(mat)
    temp=merge(mat)
    mat=temp[0]
    done=done or temp[1]
    mat=cover_up(mat)[0]
    return (mat,done)

def right(mat):   
    mat=reverse(mat)
    mat,done=cover_up(mat)
    temp=merge(mat)
    mat=temp[0]
    done=done or temp[1]
    mat=cover_up(mat)[0]
    mat=reverse(mat)
    return (mat,done)

def main():
    game = Game(4)
    game.add_two()
    game.add_two()
    E = ExpectMax()
    print(game.game_state())
    while game.game_state()=='not over':
        done = E.get_move(game)
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



if __name__ == '__main__':
    main()
    