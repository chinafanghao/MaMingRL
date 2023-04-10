import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5, 0.5, 0.5]
card_No = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
card_nums = 13
# value of face card
p_val = 0.5
# up limit value
dest = 10.5


# calculate the sum of player's cards now
def sum_card(player):
    return sum(player)


# return the number of cards in player's hand now
def get_card_num(player):
    return len(player)


# return number of face cards in player's hand now
def get_p_num(player):
    count = 0
    for card_value in player:
        if card_value == p_val:
            count += 1
    return count


# judge whether "Bust" (Bao Pai)
def get_bust(player):
    if sum_card(player) > dest:
        return True
    else:
        return False


# judge whether "Five Small" (Ren Wu Xiao )
def is_rwx(player):
    return get_card_num(player) == 5 and get_p_num(player) == 5


# judge whether "Heavenly King" (Tian Wang)
def is_tw(player):
    return get_card_num(player) == 5 and sum_card(player) == dest


# judge whether "Five Small and no Face Cards" (Wu Xiao)
def is_wx(player):
    return get_card_num(player) == 5 and get_p_num(player) == 0 and sum_card(player) < dest


# judge whether dest
def is_dest(player):
    return get_card_num(player) < 5 and sum_card(player) == dest


# judge the type of player's cards combination
def hand_types(player):
    types = 1  # default Ping Pai
    reward = 0
    done = False

    if get_bust(player):
        types = 0
        reward = -1
        done = True
    elif is_rwx(player):
        types = 5
        reward = 5
        done = True
    elif is_tw(player):
        types = 4
        reward = 4
        done = True
    elif is_wx(player):
        types = 3
        reward = 3
        done = True
    elif is_dest(player):
        types = 2
        reward = 2
        done = True
    return types, reward, done


class HalfTenEnv(gym.Env):
    '''
    "Simple Ten and a Half"
    Ten and a Half is a card game that is suitable for players of all ages. The objective of the game is to collect cards that add up to "Ten and a Half". If the total exceeds Ten and a Half, the player loses.

    In the game of Ten and a Half, the cards are as follows: Ace is worth 1 point, and the other cards are worth their face value (2 to 10). The face cards (Jack, Queen, King) are each worth half a point.

    The game is played between a banker and a player. The following are the different types of hands:

        ·"Five Small" (Ren Wu Xiao ): A hand consisting of 5 face cards. The player receives a reward of 5x the bet.
        ·"Heavenly King" (Tian Wang): A hand consisting of 5 cards that add up to exactly Ten and a Half.
                                    The player receives a reward of 4x the bet.
        ·"Five Small and no Face Cards" (Wu Xiao): A hand consisting of 5 cards that do not all have face values,
                                    and the total point value is less than Ten and a Half. The player receives a reward of 3x the bet.
        ·"Ten and a Half" (Shi Dian Ban): A hand consisting of less than 5 cards that add up to exactly Ten and a Half.
                                    The player receives a reward of 2x the bet.
        ·"Below Ten and a Half" (Ping Pai): A hand consisting of less than 5 cards that add up to less than Ten and a Half.
                                    The player receives a reward of 1x the bet.
        ·"Bust" (Bao Pai): A hand consisting of cards that add up to more than Ten and a Half.
    The hands are ranked from highest to lowest as follows: Five Small > Heavenly King > Five Small and Not All Face Cards > Ten and a Half > Below Ten and a Half > Bust.

    If a player receives a hand that is Ten and a Half or higher (including Five Small, Heavenly King, Five Small
    and Not All Face Cards, or Ten and a Half), the player wins immediately and the banker loses.
    If a player's total exceeds Ten and a Half, the player loses immediately and the banker wins.

    If a player's total is less than Ten and a Half and chooses to "stand," the banker will then draw cards and compare their total to the player's. If the banker's total is less than the player's, the banker will continue to draw cards until a winner is determined. If the banker's total is equal to the player's total, the number of cards in each hand is compared. If the banker has fewer cards than the player, the banker draws another card. If the banker has more cards than the player, the banker wins.

    The rewards for winning and losing are as follows:

    Win: +1
    Lose: -1
    When calculating the rewards, the corresponding multiplier for each hand type should be used.
    '''

    def __init__(self):
        self.player = None
        self.dealer = None
        self.cards = None
        self.action_space = spaces.Discrete(2)  # two actions: 0 is
        self.observation_space = spaces.Tuple((
            spaces.Discrete(21),  # scores
            spaces.Discrete(5),  # number of cards
            spaces.Discrete(6)  # number of face cards
        ))
        self.card_nums = card_nums
        self._seed()
        self._reset()

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        # player calls one card
        if action:
            self.player += self.draw_hand()
            types, reward, done = hand_types(self.player)
        else:
            done = True
            self.dealer = self.draw_hand()

            result = self.compare()

            if result:
                reward = -1
            else:
                while not result:
                    self.dealer += self.draw_hand()

                    dealer_types, dealer_reward, dealer_done = hand_types(self.dealer)
                    if dealer_done:
                        reward = -dealer_reward
                        break
                    result = self.compare()
                    if result:
                        reward = -1
                        break
            return self._get_obs(),reward,done,{}

    def _seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return sum_card(self.player), get_card_num(self.player), get_p_num(self.player)

    def reset(self):
        self.cards = [0] * card_nums
        self.player = self.draw_hand()
        return self._get_obs()

    def draw_card(self):
        return self.np_random.choice(card_No)

    def draw_hand(self):
        card = self.draw_card()
        while self.cards[card] >= 4:
            card = self.draw_card()
        self.cards[card] += 1
        return [deck[card]]

    def compare(self):
        if sum_card(self.dealer) > sum_card(self.player) or (
                sum_card(self.dealer) == sum_card(self.player) and get_card_num(
            self.dealer) >= get_card_num(self.player)):
            return True
        else:
            return False
