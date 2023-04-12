import sys
if "../" not in sys.path:
  sys.path.append("../")

from enviroment import HalfTen
import matplotlib.pyplot as plt
import numpy as np
from utils.MC_method import mc_control_epsilon_greedy,mc_control_weighted_importance_sampling,make_random_policy,make_episilon_greedy_policy,mc_control_unweigthed_importance_sampling
from utils.plot_func import plot_3D_HalfTen
env=HalfTen.HalfTenEnv()

content=['Bust 爆牌','Below Ten and a Half 平牌','Ten and a Half 十点半','Five Small and no Face Cards 五小','Heavenly King 天王','Five Small 人五小',]

def print_observation(observation):
    card_sum,card_num,face_num=observation
    print("player score:{},card num:{},face card num:{}".format(card_sum,card_num,face_num))

#random strategy
def random_strategy(observation):
    card_sum, card_num, face_num = observation
    return 0 if card_sum>=10 or card_num>=5 else 1

def run_random_strategy():
    for i_episode in range(20):
        observation=env.reset()
        for t in range(100):
            print_observation(observation)
            action=random_strategy(observation)
            print('will take action:{}'.format('call card' if action==1 else 'stop call'))
            observation,reward,done,_=env.step(action)
            if done:
                print_observation(observation)
                if observation[0]<=10.5:
                    print('dealer score:{}'.format(np.sum(env.dealer)))
                print("game over and reward is {}".format(reward))
                print('*'*50)
                break

def run_mc_control(env,mc_strategy,num_episodes,gamma=1.0,epsilon=0.1,alpha=0.1,max_episodes_len=1000):
    if callable(mc_strategy) and mc_strategy==mc_control_epsilon_greedy:
        Q,policy=mc_strategy(env,num_episodes,gamma,epsilon,alpha,max_episodes_len)
    elif callable(mc_strategy) and mc_strategy==mc_control_weighted_importance_sampling:
        random_policy=make_random_policy(env.action_space.n)
        Q, policy = mc_strategy(env,random_policy, num_episodes, gamma, max_episodes_len)
    elif callable(mc_strategy) and mc_strategy==mc_control_unweigthed_importance_sampling:
        Q,policy=mc_strategy(env,num_episodes,gamma,epsilon,max_episodes_len)
    policy_content=['stop call','call card']

    action_0_pcard=[]
    action_1_pcard=[]
    action_2_pcard = []
    action_3_pcard = []
    action_4_pcard = []

    result=[]

    for state,actions in Q.items():
        action_value=np.max(actions)
        best_action=np.argmax(actions)
        score,card_num,p_num=state
        item={'x':score,'y':int(card_num),'z':best_action,'p_num':p_num}
        result.append(item)

        if p_num==0: action_0_pcard.append(item)
        elif p_num==1:action_1_pcard.append(item)
        elif p_num==2:action_2_pcard.append(item)
        elif p_num==3:action_3_pcard.append(item)
        elif p_num==4:action_4_pcard.append(item)

        print('current card sum is {:.1f},card number is {}, face card num is {},best policy is {}'.format(score,card_num,p_num,policy_content[best_action]))

    plot_3D_HalfTen(action_0_pcard, '0 face card best strategy','policy')
    plot_3D_HalfTen(action_1_pcard, '1 face card best strategy', 'policy')
    plot_3D_HalfTen(action_2_pcard, '2 face card best strategy', 'policy')
    plot_3D_HalfTen(action_3_pcard, '3 face card best strategy', 'policy')
    plot_3D_HalfTen(action_4_pcard, '4 face card best strategy', 'policy')
    plt.show()

    result.sort(key=lambda obj:obj.get('x'),reverse=False)
    for temp in result:
        print('current card sum is {:.1f},card number is {},face card num is {},best policy is {}'.format(temp['x'],temp['y'],temp['p_num'],policy_content[temp['z']]))

#run_random_strategy()
#run_mc_control(env,mc_control_epsilon_greedy,100000)
#run_mc_control(env,mc_weighted_control_importance_sampling,100000)
run_mc_control(env,mc_control_unweigthed_importance_sampling,500000)