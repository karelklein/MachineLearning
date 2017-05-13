# Randomized Optimization
- Experiments using RO algorithms on Diabetes data as well as classical machine learning problems.

## Analysis
- kkc3-analysis.pdf

## Data
The Pimas Indians Diabetes dataset was divided into three portions:

- diabetes_CV.txt
- diabetes_TRAIN.txt
- diabetes_TEST.txt

TRAIN contains 60% of the data, while CV and TEST each contain 20% of the data.
To run experiments, add the above .txt files to: /ABAGAIL/src/opt/test/

## Code
Experiments were modified using the ABAGAIL library cloned from https://github.com/pushkar/ABAGAIL.
- runTests.java
- TravelingSalesmanTest.java
- FourPeaksTests.java
- NQueensTest.java

To run .java files, follow these 4 steps (replace runTests with file name):

1. Add file to: /ABAGAIL/src/opt/test/
2. Compile file: Cd to ABAGAIL and run
	- javac -cp ABAGAIL.jar src/opt/test/runTests.java
3. Add to JAR files: Cd to src and run
	- jar -uvf ../ABAGAIL.jar opt/test/runTests.class
4. Run  file:
	- java -cp ABAGAIL.jar opt.test.runTests

## Test Results
- testresults.txt
- iterationresults.txt
- hw2tests.xlsb

- testresults.txt contains the results of runTests.java run on the diabetes dataset using various hyperparameters.
- iterationresults.txt contains the results of runTests.java run on diabetes dataset using various iterations.
- hw2tests.xlsb summarizes results for the diabetes dataset experiments as well as the resulting data
for each optimization problem.

## References
- http://www.cc.gatech.edu/~isbell/tutorials/mimic-tutorial.pdf