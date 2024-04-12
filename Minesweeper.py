import numpy as np
from NN import Agent
import copy
import matplotlib.pyplot as plt

"""
Default implementation of our n-bombs minesweeper
"""


def create_board(w, h, bombs):
    board = np.zeros((w, h))

    # Place bombs
    bombs = np.random.choice(w * h, bombs, replace=False)
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
            for x in range(max(0, i - 1), min(w, i + 2)):
                for y in range(max(0, j - 1), min(h, j + 2)):
                    if board[x][y] == -1:
                        count += 1
            board[i][j] = count

    return board


def action(board, visible_board, x, y):
    if board[x][y] == -1:
        return False
    visible_board[x][y] = board[x][y]
    if board[x][y] == 0:
        for i in range(max(0, x - 1), min(len(board), x + 2)):
            for j in range(max(0, y - 1), min(len(board[0]), y + 2)):
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


def get_observation(visible_board, w, h, obs_w, obs_h, n_actions):
    """
    Get random observation from the board with at least one open tile and at least one closed tile.

    return the observation window and the x and y offset (from top left corner)
    """
    # default - observation is entire visible board
    if w == obs_w and h == obs_h:
        return visible_board, 0, 0

    # pick a random observation from the board
    assert w > obs_w and h > obs_h

    start_row = np.random.randint(0, 1 + h - obs_h)
    start_col = np.random.randint(0, 1 + w - obs_w)

    observation = visible_board[start_row:start_row + obs_h, start_col:start_col + obs_w]

    if n_actions != 0:
        # observation with at least 1 closed tile and at least 1 open tile
        closed_tiles_in_observation = np.sum(observation == -2)

        while closed_tiles_in_observation == 0 or closed_tiles_in_observation == obs_w * obs_h:
            start_row = np.random.randint(0, 1 + h - obs_h)
            start_col = np.random.randint(0, 1 + w - obs_w)
            observation = visible_board[start_row:start_row + obs_h, start_col:start_col + obs_w]
            closed_tiles_in_observation = np.sum(observation == -2)

    return observation, start_col, start_row


def choose_action(agents: list[Agent], observation):
    """
    TODO:  now, a majority vote is used to select an action.
    todo: In the case that no single action has a majority, the first is chosen
    todo: choosing the action with the highest summed Q value is probably better
    """
    if len(agents) == 1:
        curr_action = agents[0].choose_action(observation.flatten())
    else:
        actions = [agent.choose_action(observation.flatten()) for agent in agents]
        chosen_actions = []
        action_counts = []
        for a in actions:
            try:
                action_index = chosen_actions.index(a)
                action_counts[action_index] += 1
            except ValueError:
                chosen_actions.append(a)
                action_counts.append(1)
        most_chosen = np.argmax(action_counts)
        curr_action = chosen_actions[most_chosen]

    return curr_action


def play_game(w: int, h: int, agents: list[Agent], n_bombs: int,
              reward_list: list[float], trailing_reward: list[float],
              obs_w=5, obs_h=5):
    board = create_board(w, h, bombs=n_bombs)

    visible_board = np.zeros((w, h))
    visible_board.fill(-2)

    actions_taken = []
    n_actions = 0

    win = False

    done = False
    while not done and n_actions < w * h:

        observation, obs_x_offset, obs_y_offset = get_observation(visible_board, w, h, obs_w, obs_h, n_actions)

        # Choose the action with the agent
        curr_action = choose_action(agents, observation)

        # store the current state before the action
        curr_state = copy.deepcopy(observation)

        # initialize rewards
        curr_reward = 0
        reward_terminal = 0
        reward_cells = 0
        reward_new_action = 0

        # store the action taken
        actions_taken.append(curr_action)

        # get the x and y coordinates from the action
        x = curr_action // obs_w
        y = curr_action % obs_h

        x += obs_x_offset
        y += obs_y_offset

        # loss
        if not action(board, visible_board, x, y):
            done = True
            reward_terminal = -1

        # win
        if np.count_nonzero(visible_board == -2) == n_bombs:
            done = True
            win = True
            reward_terminal = 1

        if h == obs_h and w == obs_w:
            new_state = visible_board
        else:
            new_state = visible_board[obs_y_offset:obs_y_offset + obs_h, obs_x_offset:obs_x_offset + obs_w]

        if not done:
            cells_opened = 0
            # reward for each new tile opened
            for a in range(obs_w):
                for b in range(obs_h):
                    if new_state[b][a] != -2 and curr_state[b][a] == -2:
                        cells_opened += 1

            reward_cells = cells_opened / (obs_w * obs_h)

            if curr_action in actions_taken:
                reward_new_action = -1
            else:
                reward_new_action = 1

        # reward function
        curr_reward = 100 * reward_terminal + 10 * reward_cells + 25 * reward_new_action

        n_actions += 1

        # update the reward lists
        reward_list.append(curr_reward)
        trailing_reward.append(np.mean(reward_list[-1000:]))

        # store the transition
        agents[0].store_transition(curr_state.flatten(), curr_action, curr_reward, observation.flatten(), done)

    return win, n_actions


def play_games(
        n_games: int,
        w: int, h: int,
        agents: list[Agent], update_agent: bool,
        n_bombs_low=1, n_bombs_high=10,
        save_stats=False,
        obs_w=5,
        obs_h=5
):
    reward_list = []
    trailing_reward = []
    n_action_list = []
    win_ratio_list = []
    trailing_wl = []

    # winratio last 1000 games
    win_loss_ratio = np.zeros(1000)

    wins = 0

    # win rates per number of bombs
    win_rates = np.zeros(11)

    # games played per number of bombs
    games_played = np.zeros(11)

    for i in range(n_games):

        n_bombs = np.random.randint(n_bombs_low, n_bombs_high)

        # increase the number of games played for the number of bombs
        games_played[n_bombs] += 1

        win, n_actions = play_game(w, h, agents, n_bombs, reward_list, trailing_reward, obs_w, obs_h)

        if win:
            wins += 1
            win_loss_ratio[i % 1000] = 1

            # increase the winrate for the number of bombs
            win_rates[n_bombs] += 1
        else:
            win_loss_ratio[i % 1000] = 0

        # update the number of actions list
        n_action_list.append(n_actions)

        win_loss_ratio_number = np.sum(win_loss_ratio) / 1000
        win_ratio_list.append(win_loss_ratio_number)
        trailing_wl.append(np.mean(win_ratio_list[-1000:]))

        if i % 1000 == 0:
            print(
                f"game: {i} trailing_reward: {np.mean(reward_list[-1000:]):.2f} epsilon: {agents[0].epsilon:.2f}"
                f"avg_n_actions: {np.mean(n_action_list[-1000:]):.2f} wins: {wins} win_loss_ratio: {win_loss_ratio_number:.2f}")

        if i % 1000 == 0:

            # print the win rates nicely
            for j in range(11):
                gp = games_played[j]
                if gp > 0:
                    print(f"bombs: {j} winrate: {win_rates[j] / games_played[j]:.2f}")

            # reset the win rates
            win_rates = np.zeros(11)
            games_played = np.zeros(11)

        if update_agent:
            agents[0].learn()

            if i % 100_000 == 0:
                # save model with i name
                agents[0].save_model(name=f"model_{i}.pt")

        if save_stats and i % 100_000 == 0:
            save_run_stats(win_ratio_list, trailing_wl, trailing_reward)

    return win_ratio_list, trailing_wl, trailing_reward, n_action_list, wins


def save_run_stats(win_ratio_list, trailing_wl, trailing_reward):
    np.save("win_ratio.npy", np.asarray(win_ratio_list))
    np.save("trailing_wl.npy", np.asarray(trailing_wl))
    np.save("trailing_reward.npy", np.asarray(trailing_reward))

    # plot trailing wl
    plt.plot(trailing_wl)
    plt.savefig("winpercentage.png")
    plt.close()

    # plot trailing reward
    plt.plot(trailing_reward)
    plt.savefig("trailing_reward.png")
    plt.close()
