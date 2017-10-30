package weka.classifiers.trees.stgp.forest;

import weka.classifiers.trees.stgp.tree.Tree;
import weka.classifiers.trees.stgp.tree.TreeCrossoverHandler;
import weka.classifiers.trees.stgp.tree.TreeMutationHandler;
import weka.classifiers.trees.stgp.util.Mat;

public class ForestFunctions {
	/**
	 * Returns the train rmse of t
	 * @param t Tree
	 * @return 
	 * @return t's train rmse
	 */
	static double fitnessTrain(Tree t, double[][]data, double[]target, double trainFraction){
		double sum = 0;
		int trainSize = (int)(target.length * trainFraction);

		for(int i = 0; i < trainSize;i++){
			sum += Math.pow(t.getHead().calculate(data[i])-target[i], 2);
		}

		sum /= trainSize;

		return Math.sqrt(sum);
	}

	/**
	 * Returns the test rmse of t
	 * @param t Tree
	 * @return t's test rmse
	 */
	static double fitnessTest(Tree t, double[][]data, double[]target, double trainFraction){
		double sum = 0;
		int tLen = target.length;
		int trainSize = (int)(tLen * trainFraction);

		for(int i = trainSize; i < tLen ;i++){
			sum += Math.pow(t.getHead().calculate(data[i])-target[i], 2);
		}
		sum /= tLen - trainSize;

		return Math.sqrt(sum);
	}
	
	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @requires population sorted by fitness
	 * @param population Tree population
	 * @return The winner tree
	 */
	static Tree tournament(Tree [] population, int tournamentSize) {
		int pick = Mat.random(tournamentSize);
		for(int i = 1; i < tournamentSize; i ++){
			pick = Math.min(pick, Mat.random(population.length));
		}
		return population[pick];
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents of p1 and p2 through crossover and returns
	 * the smallest one
	 * @param p1 Parent 1
	 * @param p2 Parent 2
	 * @return Descendent of p1 and p2 through crossover
	 */
	static Tree crossover(Tree p1, Tree p2, int tournamentSize) {
		Tree candidate, smaller = TreeCrossoverHandler.crossover(p1, p2);
		for(int i = 1; i < tournamentSize; i++) {
			candidate = TreeCrossoverHandler.crossover(p1, p2);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents from p through mutation and returns the 
	 * smallest one
	 * @param p Original Tree
	 * @return Descendent of p by mutation
	 */
	static Tree mutation(Tree p, int tournamentSize, String [] operations, String [] terminals, double terminalRateForGrow, int maxDepth) {
		Tree candidate, smaller = TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth);
		for(int i = 1; i < tournamentSize; i++) {
			candidate = TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
}