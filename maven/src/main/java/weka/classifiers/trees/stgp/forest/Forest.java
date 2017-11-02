package weka.classifiers.trees.stgp.forest;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

import weka.classifiers.trees.stgp.client.ClientWekaSim;
import weka.classifiers.trees.stgp.tree.Tree;
import weka.classifiers.trees.stgp.tree.TreeCreationHandler;
import weka.classifiers.trees.stgp.tree.TreeCrossoverHandler;
import weka.classifiers.trees.stgp.tree.TreeMutationHandler;
import weka.classifiers.trees.stgp.util.Arrays;
import weka.classifiers.trees.stgp.util.Mat;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class Forest implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private boolean messages = true;

	// current and max generation
	private int generation = 0;
	private int maxGeneration = 10;

	// population
	private Tree [] population;

	//data, target column and fraction used for training
	private double [][] data;
	private double [] target;
	private double trainFraction;

	//fraction of the population used for tournament and elitism
	private int tournamentSize = 1;
	private int elitismSize = 0;

	//operations and terminals used
	private String [] operations;
	private String [] terminals;
	
	//probability of a node being terminal while using grow to create a tree
	private double terminalRateForGrow = 0.10;
	
	//initial max depth
	private int maxDepth = 7;

	//from the trees with the best train rmse over the generations, this is the one with the lower test rmse
	private Tree bestTree = null;

	/**
	 * Construtor
	 * @param filename
	 * @param op
	 * @param term
	 * @param maxDepth
	 * @param data
	 * @param target
	 * @param pop_size
	 * @param trainFract
	 * @throws IOException
	 */
	public Forest(String filename, String [] op, String [] term, int maxDepth, 
			double [][] data, double [] target, int pop_size, double trainFract,
			String populationType, int maxGeneration, double tournamentFraction,
			double elitismFraction) throws IOException{
		message("Creating forest...");

		this.data = data;
		this.target = target;
		this.trainFraction = trainFract;

		this.operations = op;
		this.terminals = term;
		this.maxDepth = maxDepth;

		this.maxGeneration = maxGeneration;

		tournamentSize = (int) (tournamentFraction * pop_size);
		elitismSize = (int) (elitismFraction * pop_size);
		
		population = new Tree[pop_size];

		switch(populationType) {
		case "Ramped":
			for(int i = 0; i < pop_size; i++){
				if( i < pop_size / 2)
					population[i] = TreeCreationHandler.create(op, term, 0 , Math.max(i%maxDepth,2));
				else
					population[i] = TreeCreationHandler.create(op, term, terminalRateForGrow , Math.max(i%maxDepth,2));
			}
			break;
		default: // full
			for(int i = 0; i < pop_size; i++){
				if( i < pop_size / 2)
					population[i] = TreeCreationHandler.create(op, term, 0 , maxDepth);
			}
			break;
		}		
	}

	/**
	 * Trains the classifier
	 */
	public void train() throws IOException {
		message("Starting train...");

		generation = 0;
		while(improving()){
			if(generation%5 == 0)
				message("Generation " + generation + "...");
			//TODO delete
			double [] result = nextGeneration(generation);
			ClientWekaSim.results[generation][0]+=result[0];
			ClientWekaSim.results[generation][1]+=result[1];
			ClientWekaSim.results[generation][2]++;
			generation ++;
		}
	}

	/**
	 * Evolves the classifier by one generation
	 */
	private double[] nextGeneration(int generation) throws IOException{
		double [] results = new double[2];

		Tree [] nextGen = new Tree [population.length];
		double [] fitnesses = new double[population.length];

		// Obtencao de fitness
		for (int i = 0; i < population.length; i++){
			fitnesses[i] = fitnessTrain(population[i]);
		}

		Arrays.mergeSortBy(population, fitnesses);
		
		// Elitismo
		for(int i = 0; i < elitismSize; i++ ){
			nextGen[i] = population[i];
		}
		
		//Selecao e reproducao
		Tree parent1, parent2;
		for(int i = elitismSize; i < nextGen.length; i++){
			parent1 = tournament(population);
			parent2 = tournament(population);

			if(Math.random() < 0.75) {
				nextGen[i] = crossover(parent1, parent2);
			}else {
				nextGen[i] = mutation(parent1);
			}
		}

		population = nextGen;
		
		if(elitismSize == 0) {
			setBestToFirst(population);
		}

		//Sets the bestTree to the generation best if it has a better test error 
		if(bestTree == null || fitnessTest(population[0])<fitnessTest(bestTree)) {
			bestTree = population[0];
		}
		
		results[0] = fitnessTrain(bestTree);
		results[1] = fitnessTest(bestTree);
		
		return results;
	}

	/**
	 * Sets the tree with the higher fitness to the index 0 of the population
	 * @param pop population
	 */
	private void setBestToFirst(Tree[] pop) {
		int bestIndex = 0;
		double bestRMSE = fitnessTrain(pop[0]);
		double candidateRMSE;
		for(int i = 0; i < pop.length; i++){
			candidateRMSE = fitnessTrain(pop[i]);
			if(candidateRMSE < bestRMSE){
				bestRMSE = candidateRMSE;
				bestIndex = i;
			}
		}
		Tree dup = pop[0];
		pop[0] = pop[bestIndex];
		pop[bestIndex] = dup;
	}

	/**
	 * Returns the train rmse of t
	 * @param t Tree
	 * @return t's train rmse
	 */
	private double fitnessTrain(Tree t){
		return ForestFunctions.fitnessTrain(t,data,target,trainFraction);
	}

	/**
	 * Returns the test rmse of t
	 * @param t Tree
	 * @return t's test rmse
	 */
	private double fitnessTest(Tree t){
		return ForestFunctions.fitnessTest(t,data,target,trainFraction);
	}

	/**
	 * Returns the prediction
	 * @param v arguments
	 */
	public double predict(double [] v) {
		return bestTree.getHead().calculate(v);
	}

	/**
	 * Prints a message if messages is set to true
	 * @param s
	 */
	private void message(String s){
		if(messages)
			System.out.println(s);
	}
	
	/**
	 * Returns true if the classifier is still improving
	 */
	private boolean improving() {
		return generation < maxGeneration;
	}	

	/**
	 * Returns the best tree of the train in it's String format
	 */
	public String toString(){
		if(bestTree == null)
			return "This classifier has not been trained yet.";
		else
			return bestTree.toString();
	}
	
	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @param population Tree population
	 * @return The winner tree
	 */
	private Tree tournament(Tree [] population) {
		return ForestFunctions.tournament(population, tournamentSize);
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents of p1 and p2 through crossover and returns
	 * the smallest one
	 * @param p1 Parent 1
	 * @param p2 Parent 2
	 * @return Descendent of p1 and p2 through crossover
	 */
	private Tree crossover(Tree p1, Tree p2) {
		return ForestFunctions.crossover(p1,p2, tournamentSize);
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents from p through mutation and returns the 
	 * smallest one
	 * @param p Original Tree
	 * @return Descendent of p by mutation
	 */
	private Tree mutation(Tree p) {
		return ForestFunctions.mutation(p, tournamentSize,operations, terminals, terminalRateForGrow, maxDepth);
	}
}