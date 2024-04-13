import numpy as np

import Minesweeper
from NN import Agent
import copy
import matplotlib.pyplot as plt


def main():
    n_games = 1_000

    w, h = 10, 10
    obs_w, obs_h = 5, 5
    # initialize the agent
    agent = Agent(
        gamma=0.99,
        epsilon=0.01,
        batch_size=64,
        n_actions=obs_w * obs_h,
        eps_end=0.01,
        input_dims=[obs_w * obs_h],
        lr=0.001,
        eps_dec=(1 / n_games),
        max_mem_size=100000
    )

    # initialize the agent
    agent1 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=obs_w * obs_h, eps_end=0, input_dims=[obs_w * obs_h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent2 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=obs_w * obs_h, eps_end=0, input_dims=[obs_w * obs_h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent3 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=obs_w * obs_h, eps_end=0, input_dims=[obs_w * obs_h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent4 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=obs_w * obs_h, eps_end=0, input_dims=[obs_w * obs_h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent5 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=obs_w * obs_h, eps_end=0, input_dims=[obs_w * obs_h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)

    #load 5 agents
    agent1 = agent1.load_model(name="./Model_3Bombs_5x5_batch64_lr0.001_mem100000/model_final.pt")
    agent2 = agent2.load_model(name="./Model_3Bombs_5x5_batch64_lr0.001_mem100000_eps0.75/model.pt")
    agent3 = agent3.load_model(name="./AnotherModel/model_800000.pt")
    agent4 = agent4.load_model(name="./AnotherModel2/model.pt")
    agent5 = agent5.load_model(name="./Anothermodel3/model.pt")

    agents = [agent1, agent2, agent3, agent4, agent5]

    win_ratio_list, trailing_wl, trailing_reward, _, _ = Minesweeper.play_games(n_games, w, h, agents,
                                                                                update_agent=False,
                                                                                n_bombs_low=10,
                                                                                n_bombs_high=11,
                                                                                save_stats=False,
                                                                                obs_w=obs_w,
                                                                                obs_h=obs_h)

    # save final model and stats
    Minesweeper.save_run_stats(win_ratio_list, trailing_wl, trailing_reward)

    #agent.save_model()


if __name__ == "__main__":
    main()
