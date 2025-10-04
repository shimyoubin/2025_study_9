
import numpy as np
import torch
import random
from agents.rl.submission import agent as rl_agent
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os

''' 
#기본 승 54 스텝 138
actions_map = {
    # 힘(Force) = -100 (고속 후진)
    0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
    # 힘(Force) = -40 (저속 후진)
    6: [-40, -30],  7: [-40, -18],  8: [-40, -6],  9: [-40, 6],  10: [-40, 18], 11: [-40, 30],
    # 힘(Force) = 20 (저속 전진)
    12: [20, -30],  13: [20, -18],  14: [20, -6],  15: [20, 6],  16: [20, 18],  17: [20, 30],
    # 힘(Force) = 80 (중속 전진)
    18: [80, -30],  19: [80, -18],  20: [80, -6],  21: [80, 6],  22: [80, 18],  23: [80, 30],
    # 힘(Force) = 140 (고속 전진)
    24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
    # 힘(Force) = 200 (최고속 전진)
    30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]
}
'''
'''
# actions_map_speeddemon.py 승 55 이동평균 이동: 147
actions_map = {
    # 목표: 이동율 극대화 (오직 속도)
    # 힘(Force) 설정: [-40, 20, 80, 140, 180, 200] (오직 전진)
    # 조향 각도(Angle) 설정: [-30, -20, -10, 0, 10, 20] (완벽한 직진 포함)

    # 힘(Force) = -40
    0: [-40, -30], 1: [-40, -20], 2: [-40, -10], 3: [-40, 0], 4: [-40, 10], 5: [-40, 20],
    # 힘(Force) = 20
    6: [20, -30],  7: [20, -20],  8: [20, -10],  9: [20, 0],  10: [20, 10], 11: [20, 20],
    # 힘(Force) = 80
    12: [80, -30],  13: [80, -20],  14: [80, -10],  15: [80, 0],  16: [80, 10],  17: [80, 20],
    # 힘(Force) = 140
    18: [140, -30],  19: [140, -20],  20: [140, -10],  21: [140, 0],  22: [140, 10],  23: [140, 20],
    # 힘(Force) = 180
    24: [180, -30], 25: [180, -20], 26: [180, -10], 27: [180, 0], 28: [180, 10], 29: [180, 20],
    # 힘(Force) = 200
    30: [200, -30], 31: [200, -20], 32: [200, -10], 33: [200, 0], 34: [200, 10], 35: [200, 20]
}
'''

# actions_map_smart_racer.py 승 57 스텝 132
actions_map = {
    # 목표: 공격성을 낮추고 주행 효율성 극대화 (속도와 제어의 최적 균형)
    # 힘(Force) 설정: [-60, 0, 40, 90, 140, 200] (제동 기능 부활, 속도 단계 재설계)
    # 조향 각도(Angle) 설정: [-30, -18, -8, 0, 12, 25] (직진 유지, 코너링 옵션 세분화)

    # 힘(Force) = -60 (효과적인 후진)
    0: [-60, -30], 1: [-60, -18], 2: [-60, -8], 3: [-60, 0], 4: [-60, 12], 5: [-60, 25],
    # 힘(Force) = 0 (제동 및 라인 제어)
    6: [0, -30],  7: [0, -18],  8: [0, -8],  9: [0, 0],  10: [0, 12], 11: [0, 25],
    # 힘(Force) = 40 (저속 정밀 주행)
    12: [40, -30],  13: [40, -18],  14: [40, -8],  15: [40, 0],  16: [40, 12],  17: [40, 25],
    # 힘(Force) = 90 (안정적인 순항)
    18: [90, -30],  19: [90, -18],  20: [90, -8],  21: [90, 0],  22: [90, 12],  23: [90, 25],
    # 힘(Force) = 140 (고속 주행)
    24: [140, -30], 25: [140, -18], 26: [140, -8], 27: [140, 0], 28: [140, 12], 29: [140, 25],
    # 힘(Force) = 200 (최고속 주행)
    30: [200, -30], 31: [200, -18], 32: [200, -8], 33: [200, 0], 34: [200, 12], 35: [200, 25]
}

'''
# actions_map_apex_predator.py
actions_map = {
    # 목표: '스마트 레이서' 기반, 코너 탈출 및 순항 속도를 최적화하여 스텝 수 추가 단축
    # 힘(Force) 설정: [-60, 0, 50, 100, 150, 200] (전체적인 속도 프로필 상향 조정)
    # 조향 각도(Angle) 설정: [-30, -18, -8, 0, 12, 25] (성공적이었던 설정 유지)

    # 힘(Force) = -60
    0: [-60, -30], 1: [-60, -18], 2: [-60, -8], 3: [-60, 0], 4: [-60, 12], 5: [-60, 25],
    # 힘(Force) = 0
    6: [0, -30],  7: [0, -18],  8: [0, -8],  9: [0, 0],  10: [0, 12], 11: [0, 25],
    # 힘(Force) = 50 (더 빠른 코너 탈출)
    12: [50, -30],  13: [50, -18],  14: [50, -8],  15: [50, 0],  16: [50, 12],  17: [50, 25],
    # 힘(Force) = 100 (더 높은 순항 속도)
    18: [100, -30],  19: [100, -18],  20: [100, -8],  21: [100, 0],  22: [100, 12],  23: [100, 25],
    # 힘(Force) = 150 (고속 주행)
    24: [150, -30], 25: [150, -18], 26: [150, -8], 27: [150, 0], 28: [150, 12], 29: [150, 25],
    # 힘(Force) = 200 (최고속 주행)
    30: [200, -30], 31: [200, -18], 32: [200, -8], 33: [200, 0], 34: [200, 12], 35: [200, 25]
}
'''






def get_join_actions(state, algo_list):

    joint_actions = []

    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs'].flatten()
            actions_raw = rl_agent.choose_action(obs)
            actions = actions_map[actions_raw]
            joint_actions.append([[actions[0]], [actions[1]]])

    return joint_actions






RENDER = True

def run_game(env, algo_list, episode, shuffle_map, map_num, verbose=False):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)       #agent 1 win, agent 2 win, draw
    total_steps = []            # 각 에피소드의 step 수 저장
    episode = int(episode)
    
    for i in range(1, int(episode)+1):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)
        if RENDER:
            env.env_core.render()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if RENDER:
                env.env_core.render()

            step += 1  # step 증가

            if done:
                if reward[0] != reward[1]:
                    if reward[0] == 100:
                        num_win[0] += 1
                    elif reward[1] == 100:
                        num_win[1] += 1
                        total_steps.append(step)  # 이긴 경우 에피소드별 step 수 저장
                        print(step)                        
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break
            state = next_state
        
        total_reward += episode_reward

    # 결과 출력
    total_reward /= episode
    average_steps = np.mean(total_steps) if total_steps else -1  # 평균 step 수 계산
    print("total reward: ", total_reward)
    print('Result in map {} within {} episode:'.format(map_num, episode))
    
    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
            ['win', num_win[0], num_win[1]],
            ['avg_steps', '-', average_steps]]  # 평균 step 추가
    print(tabulate(data, headers=header, tablefmt='pretty'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default='rl', help='rl/random')
    parser.add_argument("--opponent", default='random', help='rl/random')
    parser.add_argument("--episode", default=20)
    parser.add_argument("--map", default='all', help='1/2/3/4/all')
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed = 1)

    if args.map != 'all':
        game.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    #torch.manual_seed(1)
    #np.random.seed(1)
    #random.seed(1)

    agent_list = [args.opponent, args.my_ai]        #your are controlling agent green
    run_game(game, algo_list=agent_list, episode=args.episode, shuffle_map=shuffle, map_num=args.map, verbose=False)
