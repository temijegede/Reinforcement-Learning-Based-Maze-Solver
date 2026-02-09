import random
import numpy as np
from collections import deque

class Maze:

    # Define parameters for the maze
    EMPTY = ' '
    MARK = '@'
    WALL = chr(9608) # Character 9608 is '█'
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'
    cellVisited = []


    def __init__(self, width, height):
        """ This class manages a maze of the the specified width and height
        Function set includes,
        printMaze(): Visualises the maze.
        buildMaze(x, y): Builds the maze using a randomised depth first search algorithm starting from point x,y.
        braidMaze(p): Braids the maze and culls deadends. p is the probability of culling. p=1 will remove all deadends.
        deadend(): In a perfect maze, this function returns a list of all the deadends in the maze.
        """

        assert width % 2 == 1 and width >= 3 # assertion checks that size is odd and at least 3
        assert height % 2 == 1 and height >= 3

        self.width = width
        self.height = height
        self.maze = {}
        self.mazeBinaryForm = np.ones((width,height), dtype=int) # a binary form of the maze defined where WALL =.= 1 and EMPTY =.= 0
        for x in range(width):
            for y in range(height):
               self.maze[(x, y)] = self.WALL # Every space is a wall at first.


    def printMaze(self, markX=None, markY=None):
        """Displays the maze data structure in the maze argument. The
        markX and markY arguments are coordinates of the current
        '@' location of the algorithm as it generates the maze."""
         
        for y in range(self.height):
            for x in range(self.width):
                if markX == x and markY == y:
                    # if the training algorithm is here, Display the '@' mark:
                    print(self.MARK, end='')
                else:
                    # Display the wall or empty space:
                    print(self.maze[(x, y)], end='')

            print() # Print a newline after printing the row.
        print()    
        return
    
    def plotPathinMaze(self, path):
        """Displays the maze data structure in the maze argument with the selected path plotted."""
         
        for y in range(self.height):
            for x in range(self.width):
                if (y,x) in path:
                    # if the training algorithm is here, Display the '@' mark:
                    if self.maze[(x, y)] == self.EMPTY:
                        print('.', end='')
                    else:
                        print('E', end='')
                else:
                    # Display the wall or empty space:
                    print(self.maze[(x, y)], end='')

            print() # Print a newline after printing the row.
        print()    
        return


    def CarveOutCell(self,x,y,cond=None):
        """Note: to carve out entry on the boundary wall set option cond = 'entry'
          the entry point should be on the northern or western wall"""
        
        # mark entry point of maze as empty
        if x == 1 and y != 1 and cond == 'entry':
            self.maze[(0, y)] = self.EMPTY # Is entry point on the western wall
            self.mazeBinaryForm[0,y] = 0
            self.maze[(x, y)] = self.EMPTY # Carve out starting position
            self.mazeBinaryForm[x,y] = 0

        if  y == 1 and cond == 'entry':
            self.maze[(x, 0)] = self.EMPTY # Is entry point on the nothern wall
            self.mazeBinaryForm[x,0] = 0
            self.maze[(x, y)] = self.EMPTY # Carve out starting position
            self.mazeBinaryForm[x,y] = 0

        # Otherwise just carve out the requested cell.
        else:
            self.maze[(x, y)] = self.EMPTY
            self.mazeBinaryForm[x,y] = 0

        return

    def buildMaze(self,x, y):
        """"Carve out" empty spaces in the maze at x, y and then
        recursively move to neighboring unvisited spaces. This
        function backtracks when the mark has reached a dead end."""
        
        assert x % 2 == 1 # assertion checks that starting point is odd
        assert y % 2 == 1 

        # init of parameters used to track recursion
        self.cellVisited = []
        stack = deque()
        stack.append((x, y))

        self.cellVisited.append((x,y))

        self.CarveOutCell(x, y,'entry')
            

        

        while stack:
            # Check which neighboring spaces adjacent to
            # the mark have not been visited already:

            x,y = stack[-1] # top value in stack

            unvisitedNeighbors = []

            if y > 1 and (x, y - 2) not in self.cellVisited:
                unvisitedNeighbors.append(self.NORTH)

            if y < self.height - 2 and (x, y + 2) not in self.cellVisited:
                unvisitedNeighbors.append(self.SOUTH)

            if x > 1 and (x - 2, y) not in self.cellVisited:
                unvisitedNeighbors.append(self.WEST)

            if x < self.width - 2 and (x + 2, y) not in self.cellVisited:
                unvisitedNeighbors.append(self.EAST)

            if not unvisitedNeighbors:
                # BASE CASE
                # All neighboring spaces have been visited, so this is a
                # dead end. Backtrack to an earlier space:
                stack.pop()
                if not stack:
                    self.printMaze() # Display the maze
                    EndPointX = random.randrange(1, self.width-1, 2)
                    self.CarveOutCell(EndPointX, self.height-1)
                    return self.height-2, EndPointX
                continue

        
            # RECURSIVE CASE
            # Randomly pick an unvisited neighbor to visit:
            direction = random.choice(unvisitedNeighbors)

            # Move the mark to an unvisited neighboring space:

            if direction == self.NORTH:
                nextX, nextY = x, y - 2
                self.CarveOutCell(x, y - 1)

            elif direction == self.SOUTH:
                nextX, nextY = x, y + 2
                self.CarveOutCell(x, y + 1)

            elif direction == self.WEST:
                nextX, nextY = x - 2, y
                self.CarveOutCell(x - 1, y)

            elif direction == self.EAST:
                nextX, nextY = x + 2, y
                self.CarveOutCell(x + 1, y)

            self.CarveOutCell(nextX, nextY)
            
            self.cellVisited.append((nextX, nextY)) # Mark space as visited.
            stack.append((nextX, nextY))


    def __CountNeighbours(self,cell):
        x,y = cell
        neighbourCount = (self.maze[(x-1,y)] == self.EMPTY) + (self.maze[(x,y-1)] == self.EMPTY) + (self.maze[(x+1,y)] == self.EMPTY) + (self.maze[(x,y+1)] == self.EMPTY)
        return neighbourCount

    def deadend(self):
        """In a perfect maze, this function returns a list of all the deadends in the maze."""
        deadendMaze = self.maze.copy()
        deadEndArg = []

        for x in range(1,self.width-1):
            for y in range(1,self.height-1):
                if deadendMaze[(x,y)] == self.EMPTY:
                    NeighbourCount = self.__CountNeighbours((x,y))
                    if NeighbourCount == 1:
                        deadEndArg.append((x,y))
                

        return deadEndArg

    def __FindNeighbours(self, cell):
        """ find all the available neighbours of the specified cell"""
        Neighbours = []
        x,y = cell

        if y >= 3 and self.maze[(x,y-1)] != self.EMPTY:
            Neighbours.append(self.NORTH)

        if y <= self.height - 4 and self.maze[(x,y+1)] != self.EMPTY:
            Neighbours.append(self.SOUTH)

        if x >= 3 and self.maze[(x-1,y)] != self.EMPTY:
            Neighbours.append(self.WEST)

        if x <= self.width - 4 and self.maze[(x+1,y)] != self.EMPTY:
            Neighbours.append(self.EAST)

        return Neighbours


    def braidMaze(self,p):
        """Braids the maze and culls deadends. p is the probability of culling. p=1 will remove all deadends"""

        deadEndList = self.deadend()
        deadEndArg = deadEndList.copy()
        random.shuffle(deadEndArg)

        for i in range(len(deadEndArg)):
            NeighbourCount = self.__CountNeighbours(deadEndArg[i])
            if NeighbourCount > 1 or random.random() > p :
                continue
            Neighbours = self.__FindNeighbours(deadEndArg[i])

            nextIntersection = random.choice(Neighbours)
            x,y = deadEndArg[i]
            if nextIntersection == self.NORTH:
                    self.CarveOutCell(x, y - 1) # Connecting hallway.
                    

            elif nextIntersection == self.SOUTH:
                    self.CarveOutCell(x, y + 1) # Connecting hallway.
                    

            elif nextIntersection == self.WEST:
                    self.CarveOutCell(x - 1, y) # Connecting hallway.
                    

            elif nextIntersection == self.EAST:
                    self.CarveOutCell(x + 1, y) # Connecting hallway.
                    

        self.printMaze()
