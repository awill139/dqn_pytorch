import gym
import numpy as np
from utils import get_screen
from DQN import Agent
#plot learning



def main():
    env = gym.make('CartPole-v0').unwrapped

    batch_size = 128
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10

    n_actions = env.action_space.n
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    episode_durations = []


    agent = Agent(gamma = gamma, n_actions = n_actions, screen_height = screen_height,
                  screen_width = screen_width, batch_size = batch_size)

    n_games = 500
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        length = 0
        state = env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        # env.render()
        while not done:
            action = agent.select_action(state)
            _, reward, done, _ = env.step(action.item())
            last_screen = current_screen
            current_screen = get_screen(env)

            next_state = current_screen - last_screen if not done else None
            agent.store_transition(state, action, next_state, reward)
            state = next_state

            agent.learn()
            length += 1
        episode_durations.append(length)
        # plot_durations()

        if i % target_update == 0:
            agent.update_target()


    #imagine pretty plots
    print('done')
    env.close()    

if __name__ == '__main__':
    main()
    

           