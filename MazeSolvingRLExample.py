import numpy as np
import random
from Mazer import Maze

# Maze generation
startState = (1,5)
y,x = startState
genMaze = Maze(21,21)
goalState = genMaze.buildMaze(x,y)
genMaze.braidMaze(0.9)

# The binary form of the maze is required
maze = np.transpose(genMaze.mazeBinaryForm)

# define actions
actions = ["up", "down", "left", "right"]
actionMap = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

# initialise q table
qTable = np.zeros((maze.shape[0], maze.shape[1], len(actions)))

alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 300

def step(state, action):
    x, y = state
    dx, dy = actionMap[action]
    nx, ny = x + dx, y + dy

    # Check bounds and walls
    if nx < 0 or nx >= maze.shape[0] or ny < 0 or ny >= maze.shape[1] or maze[nx, ny] == 1:
        return state, -5  # hit wall
    if (nx, ny) == goalState:
        return (nx, ny), 10  # goal reached

    return (nx, ny), -1  # normal move, negative reward to encourage efficiency


for episode in range(episodes):
    state = startState
    steps = 0

    while state != goalState:
        x, y = state

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            actionIdx = random.randint(0, len(actions) - 1)
        else:
            actionIdx = np.argmax(qTable[x, y]) # find the indices of the actions that have the current max Q value

        action = actions[actionIdx]
        nextState, reward = step(state, action)
        nx, ny = nextState

        # Q-learning update
        qTable[x, y, actionIdx] += alpha * (
            reward + gamma * np.max(qTable[nx, ny]) - qTable[x, y, actionIdx]
        )

        state = nextState
        steps +=1
    continue

        
state = startState
path = [state]

while state != goalState:
    x, y = state
    actionIdx = np.argmax(qTable[x, y])
    action = actions[actionIdx]
    state, _ = step(state, action)
    path.append(state)

print("Learned path:")
print(path)
genMaze.plotPathinMaze(path)