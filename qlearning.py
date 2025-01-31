import numpy as np
import random


def printBoard(mat):
    """Affiche le plateau de jeu."""
    for row in mat:
        print(" ".join(map(str, row)))

# Init table board, where the player evolve
board = np.zeros((5, 5), dtype=int)

# The player can be in 5*5 = 25 states
# The player can take 4 actions (up, down, left, right)

# Init q-table
qTable = np.zeros((25, 4))

# Init enemies, treasures, and player
board[1, 1] = -1  # Enemy
board[2, 3] = -1  # Enemy
board[4, 4] = 1   # Treasure
# Player
board[1, 2] = 2

reward = 5
n_reward = -2
size = 5


# Check if a move is valid
def isValidMove(current_state, action, size):
    """Vérifie si une action est valide."""
    row, col = divmod(current_state, size)
    if action == 0:  # Up
        return row > 0
    elif action == 1:  # Down
        return row < size - 1
    elif action == 2:  # Left
        return col > 0
    elif action == 3:  # Right
        return col < size - 1
    return False


# Get the next state after a valid move
def next_state(current_state, action, size):
    if not isValidMove(current_state, action, size):
        return current_state

    row, col = divmod(current_state, size)
    if action == 0: # Up
        row -= 1
    elif action == 1: # Down
        row += 1
    elif action == 2: # Left
        col -= 1
    elif action == 3: # Right
        col += 1
    return row*size+col


# Q-Learning algorithm
def train(qTable, board, reward, n_reward, base_state, epochs=100, explo_rate=0.5, lr=0.1, gamma=0.9):
    for e in range(epochs):
        current_state = base_state
        done = False
        explo_rate = 0.1 * ((epochs - e)//(epochs/10))
        if (e%1000 == 0):
            print(explo_rate,e)
        while not done:
            # Exploration vs exploitation
            if random.random() < explo_rate:
                action = random.randint(0, 3)  # Random action (exploration)
            else:
                action = np.argmax(qTable[current_state])  # Best action (exploitation)

            # Calculate next state
            next_st = next_state(current_state, action, size)

            # Get reward
            row, col = divmod(next_st, size)
            if board[row, col] == 1:  # Treasure
                r = reward
                done = True
            elif board[row, col] == -1:  # Enemy
                r = n_reward
                done = True
            else:  # Neutral
                r = 0

            # Update Q-value
            qTable[current_state, action] += lr * (r + gamma * np.max(qTable[next_st]) - qTable[current_state, action])

            # Move to next state
            current_state = next_st
def pretty_print_qTable(qTable) :
    #print best action to take
    for i in range(5):
        for j in range(5) :
            action = np.argmax(qTable[i*5+j])
            if action == 0 :
                print('/\\ | ', end='')
            if action == 1 :
                print("\\/ | ", end='')
            if action == 2 :
                print(" < | ", end='')
            if action == 3 :
                print(" > | ", end='')
        print("\n------------------------")
    
    #printing de q-value of the best action to take
    for i in range(5):
       for j in range(5):
           q_values = qTable[i*5+j]
           print(f'{np.max(q_values):.2f} | ', end='')
       print("\n----------------------------------")

pretty_print_qTable(qTable)
# Train the agent
train(qTable, board, reward, n_reward, base_state=7)

# Display the Q-table
print("Q-Table après l'entraînement :")
pretty_print_qTable(qTable)

