import numpy as np
import pdb

# change the fundamental assumption for easier computation
# buy could only buy one, but sell shall sell all and then sell one.
def generate_actions_from_price_data(prices):
        max_sell = prices[-1]
        actions=[]
        for price in prices[::-1][1:]:
            if price>max_sell:
                actions.append(2)
                max_sell=price
            else:
                actions.append(1)
        actions=actions[::-1]
        port = 0
        for action in actions:
            if action ==1:
                port = port+1
            elif action ==2 and port>0:
                port = 0
            else:
                port = port -1
        return map(list,zip(action,port))

# here the prices refer to the average price of a day. the prices shall be a
# slice of all prices. Meaning that only 9 days of average prices. Note that it
# is efficient to produce the best actions by looping through all possible
# actions.
'''
def generate_actions_from_price_data(prices):
    old_profit = 0
    golden_actions = []
    for action_list in global_array:
        #the below method can also be replaces with algorithms which generates action non-iteravely but 
        #i am too lazy to do that and got lot of computation power
        profit, result_list = find_profit_from_given_action(prices, action_list)
        if profit >= old_profit:
            old_profit = profit
            golden_actions = result_list
    print(golden_actions)
    return golden_actions

def find_profit_from_given_action(prices, actions):
    portfilio = 0
    portfilio_value = 0
    result_list = []
    for index, action in enumerate(actions):
        price = prices[index]
        if action == 1: #buy
            portfilio += 1
            portfilio_value -= price
        elif action == 2: #sell
            portfilio -= 1
            portfilio_value += price
        result_list.append([action, portfilio])
    profit = portfilio_value + (portfilio) * prices[-1]
    return profit, result_list

#############################################
# the two following functions generate the possible action lists throught the
# episodes. Here episodes=9, so there are 2 to power of 9 =512 possible trail
# of actions. Notice that the assumption is in supervised data, no hold is
# necessary because the algo knows exactly what is best option.
############################################
def iteration_based_result():
    profit = 0
    #for max profit, machine don't have to hold as it knows future price it just have to sell and buy
    total_iteration_list = []
    action_list = [1,2]
    episode = 9
    global_array = []
    #pdb.set_trace();
    get_iteration_actions_recursive(action_list, [], episode, global_array)
    return global_array


def get_iteration_actions_recursive(action_list, temp_array, episode, global_array):
    #base case
    if episode == 0:
        global_array.append(temp_array)
        return global_array
    for i in action_list:
        new_temp_array = temp_array + [i]
        get_iteration_actions_recursive(action_list, new_temp_array, episode -1, global_array)
global_array = iteration_based_result()
