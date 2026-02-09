# 🧠 Tabular Q-Learning for Maze Navigation

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Q--Learning-green.svg)
![Status](https://img.shields.io/badge/status-learning%20experiment-orange.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

A clean implementation of **tabular Q-learning** applied to a procedurally generated maze.  
An agent learns to navigate from a fixed start position to a goal using only reward feedback—no prior map knowledge required.

---

## 📌 Project Overview

This project demonstrates **model-free reinforcement learning** using a discrete **Q-table**.  
The agent interacts with a maze environment, receives rewards or penalties, and incrementally improves its navigation policy.

Key features:
- Procedurally generated maze
- Discrete state and action space
- Epsilon-greedy exploration
- Classic Q-learning update rule
- Visualized learned path

---

## 🧩 Maze Environment

### Maze Generation

```python
genMaze = Maze(21, 21)
goalState = genMaze.buildMaze(x, y)
genMaze.braidMaze(0.9)
```

- Maze size: 21 × 21 (minimum of 3 x 3 and must be odd)
- The start point of the maze is selected using the values: x,y. 
- Both x and y must be odd. In the example, x = 1 so that the entry point is at the top of the maze.
- Dead ends are reduced using maze braiding which will add loops and other potetial paths to the goal.
- The goal state/exit point is randomly selected from the last row of the maze

### Maze Binary Form
For the RL implementation, the binary form of the maze is used
```python
maze = np.transpose(genMaze.mazeBinaryForm)
```
More information about the maze generation process can be found at [text](https://github.com/temijegede/Random-Maze-Generator)

## State and Action Space
### State representation
A state or position in the maze is marked by (x,y) coordinates and must be equal to 0 in the maze binary form. A position equal to 1 would be invalid as that denotes a wall

### Actions
The actions to be taken by the agent to move around the maze are defined as "up", "down", "left", "right"

```Python
actionMap = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}
```

## Q-learning Implementation
### Q-Table
In tabular Q-learning, the agent is based on a table of 'Q' values. The Q values represent the expected cumulative future reward for taking an action in a given state. The reinforcement learning algorithm updates this table while learning such that the best actions have the highest Q values after training.

### Learning Parameters
| Parameter  | Name              | Purpose                                                |
| ---------- | ----------------- | ------------------------------------------------------ |
| alpha ($` \alpha `$)   | Learning rate     | A weight that controls how much new information overrides old values                              |
| gamma  ($` \gamma `$)  | Discount factor   | Determines how much future rewards matter relative to current values                                    |
| epsilon ($`\epsilon`$) | Exploration rate  | Probability of choosing a random action rather than the action derived from the Q table               |
| episodes | Training episodes | Number of learning runs                                                          |

### Maze Environment Transition Functon
This the function that rewards the agent.
| Situation               | Reward |
| ----------------------- | ------ |
| Hit wall / invalid move | `-5`   |
| Normal move             | `-1`   |
| Reach goal              | `+10`  |

Notice there is still a small negative penalty for a normal move, this encourages the agent to be more efficient. This is important for finding the shortest path in a braided maze.

### Epsilon-Greedy Policy
This balances exploration of random actions with exploitation of best known actions recorded in the q-table.
Without exploration, the agent could converge prematurely to a suboptimal path.

### Updating the Q-Table
The standard Q-learning update rule is applied in this instance every training step.\
$` Q(s,a) \leftarrow Q(s,a) + \alpha \left[
r + \gamma \max_{a'} Q(s', a') - Q(s,a)
\right] `$
\
| Symbol         | Meaning                     |
| -------------- | --------------------------- |
| `s`            | Current state               |
| `a`            | Action taken                |
| `r`            | Immediate reward            |
| `s'`           | Next state                  |
| `max Q(s', a)` | Best future reward estimate |

### Training Loop
For each episode:
- The agent starts at the selected initial position
- Actions are chosen using epsilon-greedy strategy
- The Q-table is updated using the Q-learning update rule after each move
- The episode stops once the goal is reached

Over time, the Q-table converges to represent an optimal policy.

### Applying the learned policy

```Python
state = startState
path = [state]

while state != goalState:
    x, y = state
    actionIdx = np.argmax(qTable[x, y])
    action = actions[actionIdx]
    state, _ = step(state, action)
    path.append(state)
```
Starting at the initial entry  point of the maze, the actions with the highest Q-values are picked till the goal is reached.

## Key Limitations of This Implementation
- Q-table size grows rapidly with maze size
- It does not generalize to unseen states meaning it can only solve the maze it was trained for.
- It is not suitable for continuous states or very large maze environments

## Conclusion
This implementation is a clean and practical example of tabular Q-learning applied to maze navigation. It demonstrates how an agent can learn optimal behavior purely through interaction and reward feedback, without any prior knowledge of the environment.

For larger or more complex mazes, this approach can be extended using function approximation as was done in this project using deep Q-Networks [text](https://github.com/temijegede/Deep-Reinforcement-Learning-Based-Algorithm-for-Solving-any-Maze).



