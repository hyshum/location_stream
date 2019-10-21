from pandas import DataFrame, read_csv
from datetime import datetime
from dateutil import tz
import math
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

DATAFILE = "data/location_stream.csv"

def load_dataset(filepath=DATAFILE):
	raw_data =  read_csv(filepath, names=
		["timestamp",
		"utc_offset",
		"latitude",
		"longitude",
		"motion_state"
		])

	#Create list of datetime objects
	dt = []
	for i in range(len(raw_data)):
		dt.append(datetime.fromtimestamp(raw_data["timestamp"][i], tz.tzoffset('IST', raw_data["utc_offset"][i])))

	#Create local time series from list
	local_time_series = pd.Series(dt) 
	local_time_series = local_time_series.rename("local_time")

	#Create format data
	format_data = pd.concat([local_time_series, raw_data["latitude"], raw_data["longitude"], raw_data["motion_state"]], axis = 1)

	return format_data


#class Solution:
def determine_home_and_work(data):
		# Input: (DataFrame) Rows of timestamp/locations/motion_state
		# Output: ((Float, Float), (Float, Float)) Lat/Long of home and work
		#Assumptions:
		#The place you are stationary most frequently is home.
		#The place you are stationary 2nd most frequently is work

		# Multi-column frequency count 
		filtered_df = data[data["motion_state"] == 0]
		agg_df = filtered_df.groupby(["latitude","longitude"]).size().reset_index(name='counts')
		agg_df = agg_df.sort_values(by=["counts"], ascending=False).reset_index(drop=True)

		home_lat = agg_df.iloc[0]["latitude"]
		home_long = agg_df.iloc[0]["longitude"]
		work_lat = agg_df.iloc[1]["latitude"]
		work_long = agg_df.iloc[1]["longitude"]

		return (home_lat, home_long), (work_lat, work_long)


def meters_to_coord(meters):
	return meters/111111.0

def distance_between_coords(coord1, coord2):
	R = 6371
	delta_lat = (coord2[0] - coord1[0]) * (math.pi/180.)
	delta_long = (coord2[1] - coord1[1]) * (math.pi/180.)
	a = math.sin(delta_lat/2.) * math.sin(delta_lat/2.) + \
		math.cos(coord1[0]) * math.cos(coord2[0]) * \
		math.sin(delta_long/2.) * math.sin(delta_long/2.)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	return d * 1000
	

def fuzz_data(df,std):
	# std: standard deviations in meters
	fuzz_magnitude = meters_to_coord(std)
	df["latitude"] = [x + np.random.normal(scale=fuzz_magnitude) for x in df["latitude"]]
	df["longitude"] = [x + np.random.normal(scale=fuzz_magnitude) for x in df["longitude"]]
	return df


def evaluate(guess, truth):
	error = math.sqrt((guess[0]-truth[0])**2 + (guess[1]-truth[1])**2)
	return 111111.0 * error

def predict(data):

	X = data["latitude"].tolist()
	Y = data["longitude"].tolist()

	df = pd.DataFrame({
    'latitude': X,
    'longitude': Y
	})

	model = MeanShift().fit(df)
	centroids = model.cluster_centers_
	centroids_norm = [(x * x_std + x_mean, y * y_std + y_mean) for x,y in centroids]

	counts = {}
	for label in model.labels_:
		counts[label] = counts.get(label, 0)+1
	counts_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)

	home_guess = centroids[counts_sorted[0][0]]
	work_guess = centroids[counts_sorted[1][0]]

	return home_guess, work_guess
	

def main():
	data = load_dataset()

	home_coord, work_coord = determine_home_and_work(data)
	print "Home coords: {}".format(home_coord)
	print "Work coords: {}".format(work_coord)

	data_fuzzed = fuzz_data(data, 1000)

	data_fuzzed_filtered = data_fuzzed[data_fuzzed["motion_state"] == 0]

	home_guess, work_guess = predict(data_fuzzed_filtered)

	print "Predicted Home coords: {} (error: {} meters)".format(home_guess, distance_between_coords(home_coord,home_guess))
	print "Predicted Work coords: {} (error: {} meters)".format(work_guess, distance_between_coords(work_coord,work_guess))

if __name__ == "__main__":
	main()
