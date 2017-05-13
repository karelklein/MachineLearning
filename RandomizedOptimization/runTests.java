package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import shared.reader.ArffDataSetReader;

import java.util.*;
import java.io.*;
import java.text.*;
import java.lang.Math;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying TTT
 *
 * @author Karel Klein
 * @version 1.0
 */

public class runTests {
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static String trainingResults = "";
    private static String testResults = "";
    private static String cvResults = "";
    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void randomizedHillClimbing(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] trainData, Instance[] testData, Instance[] cvData) {
        String oaName = "RHC";
        // TODO: Check if this works
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, hiddenlayer, outputlayer});
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new RandomizedHillClimbing(nno);

        // Train the model
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, trainData); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();

        for(int j = 0; j < trainData.length; j++) {
            network.setInputValues(trainData[j].getData());
            network.run();

            predicted = Double.parseDouble(trainData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        trainingResults =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(trainingResults);

        // Test on test set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < testData.length; j++) {
            network.setInputValues(testData[j].getData());
            network.run();

            predicted = Double.parseDouble(testData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        testResults =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(testResults);
    }

    public static void simulatedAnnealing(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] trainData, Instance[] testData, Instance[] cvData, double start_temp, double cooling_rate) {
        String oaName = "SA";
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, hiddenlayer, outputlayer});
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new SimulatedAnnealing(start_temp, cooling_rate, nno);

        // Train the model
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, trainData); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();

        for(int j = 0; j < trainData.length; j++) {
            network.setInputValues(trainData[j].getData());
            network.run();

            predicted = Double.parseDouble(trainData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        trainingResults =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(trainingResults);

        // Test on cv set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < cvData.length; j++) {
            network.setInputValues(cvData[j].getData());
            network.run();

            predicted = Double.parseDouble(cvData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        cvResults =  "\nTest on CV set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(cvResults);

        // Test on test set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < testData.length; j++) {
            network.setInputValues(testData[j].getData());
            network.run();

            predicted = Double.parseDouble(testData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        testResults =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(testResults);

    }

    public static void geneticAlgorithm(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] trainData, Instance[] testData, Instance[] cvData, int populationSize, int toMate, int toMutate) {
        String oaName = "GA";
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, hiddenlayer, outputlayer});
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nno);

        // Train the model
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, trainData); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();

        for(int j = 0; j < trainData.length; j++) {
            network.setInputValues(trainData[j].getData());
            network.run();

            predicted = Double.parseDouble(trainData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        trainingResults =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(trainingResults);

        // Test on cv set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < cvData.length; j++) {
            network.setInputValues(cvData[j].getData());
            network.run();

            predicted = Double.parseDouble(cvData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        cvResults =  "\nTest on CV set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(cvResults);

        // Test on test set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < testData.length; j++) {
            network.setInputValues(testData[j].getData());
            network.run();

            predicted = Double.parseDouble(testData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        testResults =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(testResults);
    }

    public static void backPropagation(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] trainData, Instance[] testData, Instance[] cvData) {
        String Name = "BP";
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, hiddenlayer, outputlayer});
        ConvergenceTrainer trainer = new ConvergenceTrainer(
                   new BatchBackPropagationTrainer(set, network,
                       new SumOfSquaresError(), new RPROPUpdateRule()));
        // NOTE: Modified convergencetrainer to get backprop working
        // trainer.setMaxIterations(iteration);
        // Train the model
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        trainer.train();
        // train(oa, network, oaName, iteration, trainData);
        System.out.println("Convergence in "
            + trainer.getIterations() + " iterations");
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();

        for(int j = 0; j < trainData.length; j++) {
            network.setInputValues(trainData[j].getData());
            network.run();

            predicted = Double.parseDouble(trainData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        trainingResults =  "\nTest on Training set Results for " + Name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(trainingResults);

        // Test on test set
        start = System.nanoTime(); correct = 0; incorrect = 0;

        for(int j = 0; j < testData.length; j++) {
            network.setInputValues(testData[j].getData());
            network.run();

            predicted = Double.parseDouble(testData[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        testResults =  "\nTest on Test set Results for " + Name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(testResults);
    }

    // private static void learningCurve(Instance[] testData){
    // 	int[] splits = new int[]{20,30,40,50,60,70,80};
    // 	int[] trainLengths = new int[]{923,1385,1847,2309,2770,3232,3694};
    // 	int attributeLength = 166;
    // 	int inputLayer = 166, hiddenLayer = 5, outputLayer = 1;
    // 	int[] iterations = new int[]{100,200,400,600,1000,1500,2000,5000};
    // 	for(int i=0;i<splits.length;i++){
    // 		String trainPath = "src/musk_training_70_" + splits[i] + "-training.txt";
    // 		Instance[] trainData = DataParser.getData(trainPath,trainLengths[i],attributeLength);
    // 		DataSet trainSet = new DataSet(trainData);
    // 		System.out.println("--------------" + splits[i] + " % training set--------------");
    //         for(int j=0;j<iterations.length;j++){
    //         	System.out.println("--------------" + iterations[j] + " iterations--------------");
    // 	    	randomizedHillClimbing(inputLayer, hiddenLayer, outputLayer, iterations[j], trainSet, trainData, testData);
    // 	    	simulatedAnnealing(inputLayer, hiddenLayer, outputLayer, iterations[j], trainSet, trainData, testData, 1E11, 0.95);
    // 	    	geneticAlgorithm(inputLayer, hiddenLayer, outputLayer, iterations[j]/10, trainSet, trainData, testData, 50, 20, 10);
    // 	    	backPropagation(inputLayer, hiddenLayer, outputLayer, iterations[j]/10, trainSet, trainData, testData);
    //         }
    // 	}
    // }

    public static void main(String[] args) {
    	String trainPath = "/Users/Karel/ABAGAIL/src/opt/test/diabetes_TRAIN.txt";
        String testPath = "/Users/Karel/ABAGAIL/src/opt/test/diabetes_TEST.txt";
        String cvPath = "/Users/Karel/ABAGAIL/src/opt/test/diabetes_CV.txt";
    	int trainLength = 469;
        int testLength = 154;
        int cvLength = 145;
    	int attributeLength = 8;
        int classifierLength = 2;
        Instance[] trainData = initializeInstances(trainLength, attributeLength, trainPath);
        Instance[] testData = initializeInstances(testLength, attributeLength, testPath);
        Instance[] cvData = initializeInstances(cvLength, attributeLength, cvPath);
        DataSet trainSet = new DataSet(trainData);

        //Amount of nodes per layer
        int inputLayer = attributeLength;
        int hiddenLayer = 2;
        int outputLayer = 1;

        // NOTE: For now just comment out what's needed and switch between for getting optimal parameters
        int[] iterations = new int[]{500,1000};
        //int[] iterations = new int[]{10000};
        int[] hiddenLayers = new int[]{hiddenLayer};

        //double[] temperatures = new double[]{1, 5, 10, 20, 100, 500, 2500, 10000, 1E10};
        //double[] coolingRates = new double[]{0.99, 0.95, 0.9, 0.8};
        double[] temperatures = new double[]{20};
        double[] coolingRates = new double[]{0.8};

        //        int[] population = new int[]{5, 20, 30, 50, 80, 100, 200};
        int[] population = new int[]{100};
        //double[] crossover = new double[]{0.6, 0.8, 0.9, 0.95};
        double[] crossover = new double[]{0.9};
        double[] mutate = new double[]{0.2};

        for(int i=0;i<iterations.length;i++){
        	System.out.println("--------------" + iterations[i] + " iterations--------------");
        	randomizedHillClimbing(inputLayer, hiddenLayer, outputLayer, iterations[i], trainSet, trainData, testData, cvData);
            for (int j = 0; j < temperatures.length; j++) {
                for (int k = 0; k < coolingRates.length; k++) {
                    System.out.println("Here is the result for temperature " + temperatures[j] + " and cooling rate " + coolingRates[k] + "\n");
                    simulatedAnnealing(inputLayer, hiddenLayer, outputLayer, iterations[i], trainSet, trainData, testData, cvData, 10, 0.9);
                }
            }
            for (int l = 0; l < population.length; l++) {
                for (int m = 0; m < crossover.length; m++) {
                    for (int n = 0; n < mutate.length; n++) {
                        System.out.println("Here is the result for population " + population[l] + " crossover " + crossover[m] + " mutate " + mutate[n] + "\n");
                        geneticAlgorithm(inputLayer, hiddenLayer, outputLayer, iterations[i], trainSet, trainData, testData, cvData, population[l], (int) Math.ceil(population[l] * crossover[m]),  (int) Math.ceil(population[l] * mutate[n]));
                    }
                }
            }
        	backPropagation(inputLayer, hiddenLayer, outputLayer, iterations[i], trainSet, trainData, testData, cvData);
        }

        // for(int i=0;i<hiddenLayers.length;i++){
        //     System.out.println("--------------" + hiddenLayers[i] + " hidden layers--------------");
        //     for(int j=0;j<iterations.length;j++){
        //         System.out.println("--------------" + iterations[j] + " iterations--------------");
        //         randomizedHillClimbing(inputLayer, hiddenLayers[i], outputLayer, iterations[j], trainSet, trainData, testData, cvData);
        //         simulatedAnnealing(inputLayer, hiddenLayers[i], outputLayer, iterations[j], trainSet, trainData, testData, cvData, 1E11, 0.95);
        //         geneticAlgorithm(inputLayer, hiddenLayers[i], outputLayer, iterations[j]/10, trainSet, trainData, testData, cvData, 50, 20, 10);
        //         backPropagation(inputLayer, hiddenLayers[i], outputLayer, iterations[j], trainSet, trainData, testData, cvData);
        //     }
        //}
        //learningCurve(testData);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration, Instance[] data) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < iteration; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < data.length; j++) {
                network.setInputValues(data[j].getData());
                network.run();

                Instance output = data[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances(int instanceLength, int attributeLength, String path) {
        // ArffDataSetReader arffDSreader = new ArffDataSetReader(path);
        //
        // try
        // {
        //     return arffDSreader.read().getInstances();
        // }
        // catch(Exception e)
        // {
        //     e.printStackTrace();
        //     System.exit(0);
        // }
        // return null;

        /*
        NOTE:First entry is the number of instances
        Second entry is length 2, indicating attributes (if length 7 attributes is 01, 02, 03...), and
        Next entry is 10 for the actual output
        Third entry: See above
         */

         // NOTE: Change to number of instances
        double[][][] attributes = new double[instanceLength][][];

        try {
            // NOTE: Change to constructor
            BufferedReader br = new BufferedReader(new FileReader(new File(path)));

            //For each instance in the file...
            for(int i = 0; i < instanceLength; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                // NOTE: Change to attributelength
                attributes[i][0] = new double[attributeLength];
                attributes[i][1] = new double[1];

                // NOTE: Change to attribute length
                for(int j = 0; j < attributeLength; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[instanceLength];

        for(int i = 0; i < instanceLength; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // NOTE: labeled correctly for my instance set, change the note below
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
