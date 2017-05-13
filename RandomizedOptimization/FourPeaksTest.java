package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {

        int[] iterations = new int[]{1000,2000,3000,4000,5000};
        for (int j = 0; j < iterations.length; j++){
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[j]);
            double start = System.nanoTime();
            fit.train();
            double end = System.nanoTime();
            System.out.println("RHC," + iterations[j]+","+ ef.value(rhc.getOptimal()) + "," + Math.log((end-start)/Math.pow(10,9)));
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            fit = new FixedIterationTrainer(sa, iterations[j]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            System.out.println("SA," +iterations[j] +","+ ef.value(sa.getOptimal()) +"," + Math.log((end-start)/Math.pow(10,9)));
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, iterations[j]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            System.out.println("GA," +iterations[j]+"," + ef.value(ga.getOptimal())+","+Math.log((end-start)/Math.pow(10,9)));
            
            MIMIC mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, iterations[j]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            System.out.println("MIMIC,"+iterations[j]+"," + ef.value(mimic.getOptimal())+","+Math.log((end-start)/Math.pow(10,9)));
        }
    }
}
