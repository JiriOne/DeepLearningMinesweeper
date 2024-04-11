import numpy as np
from NN import Agent
import copy
import matplotlib.pyplot as plt 

def create_board(w, h, bombs):
    board = np.zeros((w,h))
    
    # Place bombs
    bombs = np.random.choice(w*h, bombs, replace=False)
    for bomb in bombs:
        x = bomb // w
        y = bomb % w
        board[x][y] = -1

    # Place numbers
    for i in range(w):
        for j in range(h):
            if board[i][j] == -1:
                continue
            count = 0
            for x in range(max(0, i-1), min(w, i+2)):
                for y in range(max(0, j-1), min(h, j+2)):
                    if board[x][y] == -1:
                        count += 1
            board[i][j] = count

    return board

def action(board, visible_board, x, y):


    if board[x][y] == -1:
        return False
    visible_board[x][y] = board[x][y]
    if board[x][y] == 0:
        for i in range(max(0, x-1), min(len(board), x+2)):
            for j in range(max(0, y-1), min(len(board[0]), y+2)):
                if visible_board[i][j] == -2:
                    action(board, visible_board, i, j)
    
    return True

def print_board(board):
    for row in board:
        for cell in row:
            if cell == -2:
                #print square
                print("■", end=" ")
            elif cell == -1:
                print("X", end=" ")
            else:
                print(int(cell), end=" ")
        print()
    

def main():

    #initialize the board
    w,h = 10,10

    init_w, init_h = 5,5
    

    reward_list = []
    trailing_reward = []
    n_action_list = []
    win_ratio_list = []
    trailing_wl = []

    #winratio last 100 games
    win_loss_ratio = np.zeros(100)

    n_games = 10000
    wins = 0

    #load the model
    agent = Agent(gamma=0.99, epsilon=0.0,batch_size=64, n_actions=init_w*init_h,eps_end=0.01,input_dims=[init_w*init_h],lr=0.001,eps_dec=0,max_mem_size=100000)
    agent = agent.load_model(name="Model_3Bombs_5x5_batch64_lr0.001_mem100000/model_final.pt")

    for i in range(n_games):

        done = False

        board = create_board(w,h,bombs=13)

        visible_board = np.zeros((w,h))
        visible_board.fill(-2)
        n_actions = 0

        actions_taken = []

        start_row = 0
        start_col = 0

        while (not done and n_actions < 25):            
            #human play
            #x = int(input("Enter x: "))
            #y = int(input("Enter y: "))
            # print_board(visible_board)

            # next = input("go next?")


            #grab random 5x5 area from visible board
            # Generate random indices for the starting point of the 5x5 area

            start_row = np.random.randint(0, 6)  # Random starting row index between 0 and 5 (inclusive)
            start_col = np.random.randint(0, 6)  # Random starting column index between 0 and 5 (inclusive)

            

                

            # Slice the 5x5 area from the original array
            random_5x5_area = visible_board[start_row:start_row+5, start_col:start_col+5]

            if n_actions > 0:
                while np.count_nonzero(random_5x5_area == -2) == 0:
                    start_row = np.random.randint(0, 6)
                    start_col = np.random.randint(0, 6)
                    random_5x5_area = visible_board[start_row:start_row+5, start_col:start_col+5]

            curr_action = agent.choose_action(random_5x5_area.flatten()) 

            actions_taken.append(curr_action) 

            #adjust the current action to the correct position on the board
            curr_action = curr_action + start_row * w + start_col
         
            x = curr_action // w
            y = curr_action % w

            #print("x: ", x, "y: ", y)

            #loss
            if not action(board, visible_board, x, y):
                done = True   
                win_loss_ratio[i % 100] = 0
                #print("loss")

            #win
            if np.count_nonzero(visible_board == -2) == 3:
                done = True
                wins += 1
                win_loss_ratio[i % 100] = 1    

                #print("win")

            n_actions += 1

                  
        
        n_action_list.append(n_actions)       

        win_loss_ratio_number = np.sum(win_loss_ratio) / 100
        win_ratio_list.append(win_loss_ratio_number)
        trailing_wl.append(np.mean(win_ratio_list[-1000:]))

        if i % 100 == 0:
            
            print(f"game: {i} trailing_reward: {np.mean(reward_list[-100:]):.2f} epsilon: {agent.epsilon:.2f} actions: {n_actions} wins: {wins} win_loss_ratio: {win_loss_ratio_number:.2f}")

    #plot trailing reward
    plt.plot(trailing_wl)
    plt.savefig("winpercentage_10x10.png")
    
    #agent.save_model()



if __name__ == "__main__":
    main()