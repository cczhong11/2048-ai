'''
this file is for game class
'''
from random import *

class Game(object):
    
    def __init__(self, n):
        self.grid = [[0]*n for _ in range(n)]
        self.length = n
        self.score = 0
        pass

    def game_state(self):
        for i in range(self.length):
            for j in range(self.length):
                if self.grid[i][j]==2048:
                    return 'win'
        for i in range(self.length-1): #intentionally reduced to check the row on the right and below
            for j in range(self.length-1): #more elegant to use exceptions but most likely this will be their solution
                if self.grid[i][j]==self.grid[i+1][j] or self.grid[i][j+1]==self.grid[i][j]:
                    return 'not over'
        for i in range(self.length): #check for any zero entries
            for j in range(self.length):
                if self.grid[i][j]==0:
                    return 'not over'
        for k in range(self.length-1): #to check the left/right entries on the last row
            if self.grid[self.length-1][k]==self.grid[self.length-1][k+1]:
                return 'not over'
        for j in range(self.length-1): #check up/down entries on last column
            if self.grid[j][self.length-1]==self.grid[j+1][self.length-1]:
                return 'not over'
        return 'lose'


    def add_two(self):
        ''' add random two in the grid in initialize'''
        random_x =randint(0,len(self.grid)-1)
        random_y =randint(0,len(self.grid)-1)
        while(self.grid[random_x][random_y]!=0):
            random_x=randint(0,len(self.grid)-1)
            random_y=randint(0,len(self.grid)-1)
        self.grid[random_x][random_y]=2
    
    def reverse(self):
        '''reverse grid for merge'''
        new=[]
        for i in range(self.length):
            new.append([])
            for j in range(self.length):
                new[i].append(self.grid[i][self.length-j-1])
        self.grid = new
    
    def transpose(self):
        '''transport grid for merge'''
        new=[]
        for i in range(self.length):
            new.append([])
            for j in range(self.length):
                new[i].append(self.grid[j][i])
        self.grid = new

    def cover_up(self):
        new=[[0]*self.length for _ in range(self.length)]
        done=False
        for i in range(self.length):
            count=0
            for j in range(self.length):
                if self.grid[i][j]!=0:
                    new[i][count]=self.grid[i][j]
                    if j!=count:
                        done=True
                    count+=1
        self.grid = new
        return done

    def merge(self):
        done=False
        for i in range(self.length):
            for j in range(self.length-1):
                if self.grid[i][j]==self.grid[i][j+1] and self.grid[i][j]!=0:
                    self.grid[i][j]*=2
                    self.score += self.grid[i][j]
                    self.grid[i][j+1]=0
                    done=True
        return done

    def up(self):
        print("up")
        # return self.gridrix after shifting up
        self.transpose()
        done1 = self.cover_up()
        done2 = self.merge()
        done= done1 or done2
        self.cover_up()
        self.transpose()
        return done

    def down(self):
        print("down")
        # return self.gridrix after shifting up
        self.reverse(self.transpose())
        done1 = self.cover_up()
        done2 = self.merge()
        done= done1 or done2
        self.cover_up()
        self.transpose(self.reverse())
        return done

    def left(self):
        print("left")
        # return self.gridrix after shifting up
        done1 = self.cover_up()
        done2 = self.merge()
        done= done1 or done2
        self.cover_up()
        return done

    def right(self):
        print("rigth")
        # return self.gridrix after shifting up
        self.reverse()
        done1 = self.cover_up()
        done2 = self.merge()
        done= done1 or done2
        self.cover_up()
        self.reverse()
        return done