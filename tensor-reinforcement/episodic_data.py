import gzip
import os

import numpy as np
import six
from six.moves.urllib import request

from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np
import dateutil.parser
import pdb
import glob
import cPickle as pickle
import shelve
import six
from six.moves.urllib import request
import hashlib
import json
from supervised_helper import generate_actions_from_price_data


episode = 10 #length of one episode
data_array = []
parent_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
raw_data_file  = os.path.join(parent_dir,'tensor-reinforcement/NIFTY50.csv') 
moving_average_number = 1000 #number of time interval for calculating moving average
#pdb.set_trace()

# this function read in the original raw data, and smoothen them with 1000
# days. the new data is the averaged [last_min_high+low+close+count+average,
# last_30_min_high+low+close+count+average,last_day,last_30_days]
# My personal guess is that this is the states to feed in.
# data.pkl: the new data described above
# data_dict.pkl: {key:value} the key is the compressed form of state descibed
# above, value is the real data at the current timestamp
# data_dict: {state:time point data}
def prepare_data():
	stock_data = genfromtxt(raw_data_file, delimiter=',', dtype=None, names=True)
	average_dataset = []
	total_data = []
	temp_episode = []
	data_dict = {}
	index = 0
	for data in stock_data:
		temp = [data[2], data[3], data[4], data[5],data[8]]
                # open, high, low, close, count
		average_dataset.append(temp)
		print(index)
		print(len(average_dataset))
		if index > moving_average_number:
			mean = find_average(average_dataset)
			mean_array = average_dataset/mean
                        # normalise the arrays by their mean
			last_one_hour_average = find_average(mean_array[-60:])
			last_one_day_average = find_average(mean_array[-300:])
			last_3_day_average = find_average(mean_array[-900:]) #this might change
			last_minute_data = mean_array[-1]
			average_dataset = average_dataset[1:]
			vector = []
			vector.extend(last_minute_data)
			vector.extend(last_one_hour_average)
			vector.extend(last_one_day_average)
			vector.extend(last_3_day_average)
                        # vector is [1*12 list]
			average_price = sum(temp[0:-2]) / float(len(temp[0:-2]))
			dict_vector = temp + [average_price]
                        # open, high. low, close, count, average
			data_dict[list_md5_string_value(vector)] = dict_vector
			total_data.append(vector)
		index += 1
	with open("data.pkl", "wb") as myFile:
		six.moves.cPickle.dump(total_data, myFile, -1)
	print("Done")
	with open("data_dict.pkl","wb") as myFile:
		six.moves.cPickle.dump(data_dict, myFile, -1)

def find_average(data):
    return np.mean(data, axis=0)

#############################################################################################################
# these two functions read in the original averaged data and then pack them
# into episodes. the trick is in zip(*[iter(data)]*episode): it cuts the (data)
# into sections with length of "episodes".
#
# data is an n*12 list matrix. after this it becomes floor(n/10)*10*12 list
# matrix. each real episode that this funtion refers to is the 10*12 list
# matrix, namely 10 states in a row
#
# map(list) only wrap a pair of bracket around the result of the upper stuff.
# 
def load_data(file,episode):
	data = load_file_data(file)
	return map(list,zip(*[iter(data)]*episode))

def load_file_data(file):
	with open(file, 'rb') as myFile:
		data = six.moves.cPickle.load(myFile)
	return data


# encode a list by md5 encoding methods, and then transform it to a  key
# resented with hexidexiaml value
def list_md5_string_value(list):
	string = json.dumps(list)
	return hashlib.md5(string).hexdigest()
##############################################################################################################

# the following three functions are the next group of silly functions.
# I wonder why the writer wants to put everything in a tiny piece of function
# calling the function takes mememory and computational power.
def episode_supervised_data(data, data_dict):
	prices = []
	for iteration in data:
                # iteration is a state
		prices.append(data_average_price(data_dict, iteration))
        # prices is a series of average price
	actions = generate_actions_from_price_data(prices)
        # actions should be the target policy actions awaiting to be updated
	return actions

# here the data refers to one single observed state(last_min,
# last_30_min,last_day,last_30_days)
# the function simply extracts the average price of one single time points
def data_average_price(data_dict, data):
	data =  data_dict[list_md5_string_value(data)]
	return data[-1]

# input is the floor(n/10)*10*12 list matrix. and data_dict.pkl described above
# output is [actions_for_a_episode]*floor(n/10)
def make_supervised_data(data, data_dict):
	supervised_data = []
	if not os.path.exists('supervised_data.pkl'):
		for episode in data:
                        # each episode is a series of state
			supervised_data.append(episode_supervised_data(episode, data_dict))
		with open("supervised_data.pkl", "wb") as myFile:
			six.moves.cPickle.dump(supervised_data, myFile, -1)
	with open('supervised_data.pkl', 'rb') as myFile:
		supervised_data = six.moves.cPickle.load(myFile)
	return supervised_data
#prepare_data()


