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
                print("?", end=" ")
            elif cell == -1:
                print("X", end=" ")
            else:
                print(int(cell), end=" ")
        print()
    

def main():

    #initialize the board
    w,h = 5,5

    reward_list = []
    trailing_reward = []
    n_action_list = []
    win_ratio_list = []
    trailing_wl = []

    #winratio last 100 games
    win_loss_ratio = np.zeros(100)

    n_games = 1_000_000
    wins = 0

    #winarates per number of bombs
    win_rates = np.zeros(11)
    
    #games played per number of bombs
    games_played = np.zeros(11)

    #winrates over time per number of bombs
    win_rates_time = np.zeros(11)

    #initialize the agent
    agent = Agent(gamma=0.99, epsilon=1.0,batch_size=64, n_actions=w*h,eps_end=0.01,input_dims=[w*h],lr=0.001,eps_dec=(1/n_games),max_mem_size=100000)

    #load the model
    # agent = Agent(gamma=0.99, epsilon=0.0,batch_size=64, n_actions=w*h,eps_end=0.01,input_dims=[w*h],lr=0.04,eps_dec=(1/n_games),max_mem_size=50000)
    # agent = agent.load_model(name="model_300000.pt")

    for i in range(n_games):

        done = False

        n_bombs = np.random.randint(1, 10)
        board = create_board(w,h,bombs=n_bombs)

        #increase the number of games played for the number of bombs
        games_played[n_bombs] += 1

        visible_board = np.zeros((w,h))
        visible_board.fill(-2)
        n_actions = 0

        actions_taken = []

        while (not done and n_actions < 25):            
            #print_board(visible_board)
            #human play
            #x = int(input("Enter x: "))
            #y = int(input("Enter y: "))

            #next = input("go next?")

            #Choose the action with the agent
            curr_action = agent.choose_action(visible_board.flatten())

            #store the current state before the action
            curr_state = copy.deepcopy(visible_board)

            #initialize rewards
            curr_reward = 0
            reward_terminal = 0
            reward_cells = 0
            reward_new_action = 0

            #store the action taken
            actions_taken.append(curr_action)
         
            #get the x and y coordinates from the action
            x = curr_action // w
            y = curr_action % w

            #print("x: ", x, "y: ", y)

            #loss
            if not action(board, visible_board, x, y):
                done = True   
                reward_terminal = -1
                win_loss_ratio[i % 100] = 0

            #win
            if np.count_nonzero(visible_board == -2) == n_bombs:
                done = True
                wins += 1
                reward_terminal = 1
                win_loss_ratio[i % 100] = 1   

                #increase the winrate for the number of bombs
                win_rates[n_bombs] += 1

            
            if not done:
                cells_opened = 0
                #reward for each new tile opened
                for a in range(w):
                    for b in range(h):
                        if visible_board[a][b] != -2 and curr_state[a][b] == -2:
                            cells_opened += 1
            
                reward_cells = cells_opened / (w*h)

                if curr_action in actions_taken:
                    reward_new_action = -1
                else:
                    reward_new_action = 1

            #reward function
            curr_reward = 100*reward_terminal + 10*reward_cells + 25*reward_new_action

            n_actions += 1
            
            #store the transition
            agent.store_transition(curr_state.flatten(), curr_action, curr_reward, visible_board.flatten(), done)  

            #update the reward lists
            reward_list.append(curr_reward)
            trailing_reward.append(np.mean(reward_list[-1000:]))

                  
        #update the number of actions list
        n_action_list.append(n_actions)       

        agent.learn()
        win_loss_ratio_number = np.sum(win_loss_ratio) / 100
        win_ratio_list.append(win_loss_ratio_number)
        trailing_wl.append(np.mean(win_ratio_list[-1000:]))

        if i % 100_000 == 0:
            #save model with i name
            agent.save_model(name=f"model_{i}.pt")

        if i % 1000 == 0:

            #print the win rates nicely
            print("Win rates: ")
            for j in range(11):
                print(f"bombs: {j} winrate: {win_rates[j] / games_played[j]:.2f}")

            print("Win rates over time: ")
            print(win_rates_time)

            #reset the win rates
            win_rates = np.zeros(11)
            games_played = np.zeros(11)

        if i % 100 == 0:
            
            print(f"game: {i} trailing_reward: {np.mean(reward_list[-100:]):.2f} epsilon: {agent.epsilon:.2f} actions: {n_actions} wins: {wins} win_loss_ratio: {win_loss_ratio_number:.2f}")

        
    
    #plot trailing reward
    plt.plot(trailing_wl)
    plt.savefig("winpercentage.png")
    plt.close()

    plt.plot(trailing_reward)
    plt.savefig("trailing_reward.png")
    plt.close()
    
    agent.save_model()



if __name__ == "__main__":
    main()