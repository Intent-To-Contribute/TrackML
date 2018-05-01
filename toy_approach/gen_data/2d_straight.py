import sys
import csv
import random
import math

def generate_track(track_id):
	track = []
	num_sensors = 10
	theta = random.uniform(0, 2*math.pi)
	for i in range(1, num_sensors+1):
		x = i*math.cos(theta)
		y = i*math.sin(theta)
		track.append({
			"track_id": track_id,
			"seq_id": 	i,
			"x":		x,
			"y":		y
		})

	return track

def main():
	if len(sys.argv) > 1:
		print "Generating " + sys.argv[1] + " tracks"
		num_tracks = int(sys.argv[1])
	else:
		print "Generating 1000 tracks"
		num_tracks = 1000

	with open('2d_straight.csv', 'wb') as csvfile:
		fieldnames = ['track_id', 'seq_id', 'x', 'y']

		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
		writer.writeheader()

		for i in range(0, num_tracks):
			if i % 100 == 0: print "Generating tracks " + str(i) + "-" + str(i+99) + "..."
			track = generate_track(i)

			for point in track:
				writer.writerow(point);
	# print generate_track(0)

if __name__ == "__main__":
	main()