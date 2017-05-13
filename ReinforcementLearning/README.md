# Reinforcement Learning
Experiments with RL in a Grid World environment and analysis of performance.

## Analysis
- kkc3-analysis.pdf

## Code
The java library to run MDP experiments is based on Burlap and is cloned from https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4.

- To compile the java library, run via command line:
	- mvn compile

- To perform all experiments, run via command line:
	- mvn exec:java -e -Dexec.mainClass="assignment4.EasyGridWorldLauncher" > output.txt

- To produce all graphs and plots, run the python script via command line:
	- python plot_results.py output.txt

## Dependencies
- Java
- Maven
- Burlap
- Python 2.7
- matplotlib
- numpy

## References
- "Machine Learning" by Tom Mitchell, 1997
- http://uhaweb.hartford.edu/compsci/ccli/projects/QLearning.pdf
