package weka.classifiers.trees.stgp.population;

import weka.classifiers.trees.stgp.tree.Tree;
import weka.classifiers.trees.stgp.tree.TreeCrossoverHandler;
import weka.classifiers.trees.stgp.tree.TreeMutationHandler;
import weka.classifiers.trees.stgp.util.Mat;

public class PopulationFunctions {
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
	static boolean smallerIsBetter = true;
	static Tree tournament(Tree [] population, int tournamentSize) {
		int pick = Mat.random(tournamentSize);
		for(int i = 1; i < tournamentSize; i ++){
			if(smallerIsBetter)
				pick = Math.min(pick, Mat.random(population.length));
			else
				pick = Math.max(pick, Mat.random(population.length));
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
	static Tree[] crossover(Tree[] population, int tournamentSize) {
		Tree[] parents = new Tree[2];
		Tree[] candidates = new Tree[2];
		int best = 0;
		double d =  Math.random()/2+0.5;
		for(int p = 0; p < parents.length; p++) {
			for(int i = 0 ; i < candidates.length;i++) {
				candidates[i] = tournament(population, tournamentSize);
			}
			best=candidates[1].getSize()> candidates[0].getSize()?1:0;
			if(Math.random()<d)
				parents[p]=candidates[best];
			else
				parents[p]= candidates[best==1?0:1];
		}
		return TreeCrossoverHandler.crossover(parents[0], parents[1]);
		
		/*Tree candidate, smaller = TreeCrossoverHandler.crossover(p1, p2);
		for(int i = 1; i < tournamentSize; i++) {
			candidate = TreeCrossoverHandler.crossover(p1, p2);
			if(candidate.getSize() < smaller.getSize()) {
				smaller = candidate;
			}
		}
		return smaller;*/
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents from p through mutation and returns the 
	 * smallest one
	 * @param p Original Tree
	 * @return Descendent of p by mutation
	 */
	static Tree mutation(Tree[] population, int tournamentSize, String [] operations, String [] terminals, double terminalRateForGrow, int maxDepth) {
		Tree parent = tournament(population, tournamentSize);
		Tree son = TreeMutationHandler.mutation(parent, operations, terminals, terminalRateForGrow, maxDepth);
		/*for(int i = 1; i < tournamentSize; i++) {
			candidate = TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth);
			if(candidate.getSize() < smaller.getSize()) {
				smaller = candidate;
			}
		}*/
		return son;
	}
}