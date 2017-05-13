import matplotlib.pyplot as plt
import numpy as np
import sys

def getfile(file):
	lines = []
	with open(file,'r') as f:
		for line in f:
			 lines.append(line.strip('\n').split(','))
	return lines

def getValues(file):
	lines = getfile(file)
	c = 0
	for line in lines:
		# iterations to reach convergence
		if 'terminal' in line[0]:
			iterations = [int(x) for x in lines[c + 1] if 'Iterations' not in x]
			viConverge = [int(x) for x in lines[c + 2] if 'Value' not in x]
			piConverge = [int(x) for x in lines[c + 3] if 'Policy' not in x]
			qConverge = [int(x) for x in lines[c + 4] if 'Learning' not in x]

		# time to reach convergence
		elif 'milliseconds' in line[0]:
			viTime = [int(x) for x in lines[c + 3] if 'Value' not in x]
			piTime = [int(x) for x in lines[c + 4] if 'Policy' not in x]
			qTime = [int(x) for x in lines[c + 5] if 'Q' not in x]

		# reward
		elif 'reward' in line[0]:
			viReward = [float(x) for x in lines[c + 3] if 'Value' not in x]
			piReward = [float(x) for x in lines[c + 4] if 'Policy' not in x]
			qRewards = [float(x) for x in lines[c + 5] if 'Q' not in x]
		c += 1

	steps2converge = [viConverge, piConverge, qConverge]
	time2converge = [viTime, piTime, qTime]
	rewards = [viReward, piReward, qRewards]
	return iterations, steps2converge, time2converge, rewards

def cumSteps(arr):
	# cumulative steps
	return [[sum(steps_arr[:i + 1]) for i in range(len(steps_arr))] for steps_arr in arr]

def avgRewards(rewards, steps):
	viavg, piavg, qavg = [], [], []
	for i in range(len(rewards[0])):
		viavg.append(float(rewards[0][i]) / steps[0][i])
		piavg.append(float(rewards[1][i]) / steps[1][i])
		qavg.append(float(rewards[2][i]) / steps[2][i])
	return viavg, piavg, qavg

def plot_plot(iterations, yvalues, ylabel, title, legend_loc):
	it = iterations
	vi, pi, q = yvalues
	maxim = max(max(vi), max(pi), max(q))
	minim = min(min(vi), min(pi), min(q))

	plt.plot(iterations, vi, label='Value Iteration', linewidth=2, color='blue')
	plt.plot(iterations, pi, label='Policy Iteration', linewidth=2, color='orange')
	plt.plot(iterations, q, label='Q Learning', linewidth=2, color='teal', alpha=0.55)

	plt.xticks(np.arange(0, max(it), len(it)/20))
	#plt.yticks(np.arange(-500, 80, 50))
	#plt.ylim(ymax=250)

	plt.grid(True)
	plt.legend(loc=legend_loc)
	plt.xlabel('Number of Iterations')
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def main():
	filename = sys.argv[1]
	iters, steps, time, rewards = getValues(filename)
	cum = cumSteps(steps)
	avgreward = avgRewards(rewards, steps)

	plot_plot(iters, avgreward, 'Avg Reward', 'Average Reward: Hard', 'upper left')
	plot_plot(iters, steps, 'Steps to Convergence', 'Steps to Goal: Easy', 'upper right')
	plot_plot(iters, cum, 'Cumulative Steps', 'Cumulative Steps: Hard','center right')
	plot_plot(iters, time, 'Time (ms)', 'Time to Calculate Policy: Easy', 'upper left')
	plot_plot(iters, rewards, 'Reward', 'Reward: Easy', 'lower right')

if __name__ == '__main__':
	main()