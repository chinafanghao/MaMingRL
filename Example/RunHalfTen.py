from enviroment import HalfTen
import numpy as np
env=HalfTen.HalfTenEnv()

content=['Bust 爆牌','Below Ten and a Half 平牌','Ten and a Half 十点半','Five Small and no Face Cards 五小','Heavenly King 天王','Five Small 人五小',]

def print_observation(observation):
    card_sum,card_num,face_num=observation
    print("player score:{},card num:{},face card num:{}".format(card_sum,card_num,face_num))

#random strategy
def random_strategy(observation):
    card_sum, card_num, face_num = observation
    return 0 if card_sum>=10 or card_num>=5 else 1

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
