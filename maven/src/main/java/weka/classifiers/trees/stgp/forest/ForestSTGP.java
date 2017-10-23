package weka.classifiers.trees.stgp.forest;

import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.trees.stgp.client.ClientWekaSim;
import weka.classifiers.trees.stgp.tree.TreeSTGP;
import weka.classifiers.trees.stgp.tree.TreeSTGPCrossoverHandler;
import weka.classifiers.trees.stgp.tree.TreeSTGPMutationHandler;
import weka.classifiers.trees.stgp.util.Arrays;
import weka.classifiers.trees.stgp.util.Mat;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class ForestSTGP implements Forest{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private boolean messages = true;

	// current and max generation
	private int generation = 0;
	private int maxGeneration = 10;

	// population
	private TreeSTGP [] population;

	//data, target column and fraction used for training
	private double [][] data;
	private double [] target;
	private double trainFraction;

	//fraction of the population used for tournament and elitism
	private double tournamentFraction = 0.04;
	private double elitismFraction = 0.01;

	//operations and terminals used
	private String [] operations;
	private String [] terminals;
	
	//probability of a node being terminal while using grow to create a tree
	private double terminalRateForGrow = 0.10;
	
	//initial max depth
	private int maxDepth = 7;

	//from the trees with the best train rmse over the generations, this is the one with the lower test rmse
	private TreeSTGP bestTree = null;

	/**
	 * Construtor
	 * @param filename
	 * @param op
	 * @param term
	 * @param maxDepth
	 * @param t_rate
	 * @param data
	 * @param target
	 * @param pop_size
	 * @param trainFract
	 * @throws IOException
	 */
	public ForestSTGP(String filename, String [] op, String [] term, int maxDepth, 
			double [][] data, double [] target, int pop_size, double trainFract,
			String populationType, int maxGeneration) throws IOException{
		message("Creating forest...");

		this.data = data;
		this.target = target;
		this.trainFraction = trainFract;

		this.operations = op;
		this.terminals = term;
		this.maxDepth = maxDepth;

		this.maxGeneration = maxGeneration;

		population = new TreeSTGP[pop_size];

		switch(populationType) {
		case "Ramped":
			for(int i = 0; i < pop_size; i++){
				if( i < pop_size / 2)
					population[i] = new TreeSTGP(op, term, 0 , Math.max(i%maxDepth,2));
				else
					population[i] = new TreeSTGP(op, term, terminalRateForGrow , Math.max(i%maxDepth,2));
			}
			break;
		default: // full
			for(int i = 0; i < pop_size; i++){
				if( i < pop_size / 2)
					population[i] = new TreeSTGP(op, term, 0 , maxDepth);
			}
			break;
		}
		
		TreeSTGP.trainSize = (int)(target.length * trainFraction);

	}

	/**
	 * Sets the fraction of the population used in tournaments to d
	 */
	public void setTournamentFraction(double d){
		tournamentFraction = d;
	}
	
	/**
	 * Sets the fraction of the population selected by elitism to d
	 */
	public void setElitismFraction(double d){
		elitismFraction = d;
	}

	/**
	 * Trains the classifier
	 */
	public ArrayList<Double>[] train() throws IOException {
		message("Starting train...");

		generation = 0;
		while(improving()){
			if(generation%5 == 0)
				message("Generation " + generation + "...");
			//TODO delete
			double [] result = nextGeneration(generation);
			//ClientWekaSim.results[generation][0]+=result[0];
			//ClientWekaSim.results[generation][1]+=result[1];
			//ClientWekaSim.results[generation][2]++;
			generation ++;
		}
		return null;
	}

	/**
	 * Returns true if the classifier is still improving
	 */
	public boolean improving() {
		return generation < maxGeneration;
	}	

	/**
	 * Evolves the classifier by one generation
	 */
	public double[] nextGeneration(int generation) throws IOException{
		double [] results = new double[2];

		TreeSTGP [] nextGen = new TreeSTGP [population.length];
		double [] fitnesses = new double[population.length];

		// Obtencao de fitness
		for (int i = 0; i < population.length; i++){
			fitnesses[i] = rmseTrain(population[i]);
		}

		Arrays.mergeSortBy(population, fitnesses);
		
		// Elitismo
		for(int i = 0; i < getElitismSize(); i++ ){
			nextGen[i] = population[i];
		}
		
		//Selecao e reproducao
		TreeSTGP parent1, parent2;
		for(int i = getElitismSize(); i < nextGen.length; i++){
			parent1 = tournament(population);
			parent2 = tournament(population);

			if(Math.random() < 0.75) {
				nextGen[i] = crossover(parent1, parent2);
			}else {
				nextGen[i] = mutation(parent1);
			}
		}

		results[0] = rmseTrain(population[0]);
		results[1] = rmseTest(population[0]);

		population = nextGen;
		
		if(getElitismSize() == 0) {
			setBestToFirst(population);
		}

		//Sets the bestTree to the generation best if it has a better test error 
		if(bestTree == null || rmseTest(population[0])<rmseTest(bestTree)) {
			bestTree = population[0];
		}
		
		return results;
	}

	/**
	 * Sets the tree with the higher fitness to the index 0 of the population
	 * @param pop population
	 */
	private void setBestToFirst(TreeSTGP[] pop) {
		int bestIndex = 0;
		double bestRMSE = rmseTrain(pop[0]);
		double candidateRMSE;
		for(int i = 0; i < pop.length; i++){
			candidateRMSE = rmseTrain(pop[i]);
			if(candidateRMSE < bestRMSE){
				bestRMSE = candidateRMSE;
				bestIndex = i;
			}
		}
		TreeSTGP dup = pop[0];
		pop[0] = pop[bestIndex];
		pop[bestIndex] = dup;
	}

	/**
	 * Returns the tournament absolute size
	 * @return number of individuals on the tournament
	 */
	private int getTournamentSize(){
		return (int) (tournamentFraction * population.length);
	}

	/**
	 * Returns the elite absolute size
	 * @return number of individuals on the elite
	 */
	private int getElitismSize(){
		return (int) (elitismFraction * population.length);
	}

	/**
	 * Returns the train rmse of t
	 * @param t Tree
	 * @return t's train rmse
	 */
	private double rmseTrain(TreeSTGP t){
		return t.getTrainRMSE(data,target,trainFraction);
	}

	/**
	 * Returns the test rmse of t
	 * @param t Tree
	 * @return t's test rmse
	 */
	private double rmseTest(TreeSTGP t){
		return t.getTestRMSE(data,target,trainFraction);
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
	private TreeSTGP tournament(TreeSTGP [] population) {
		int tournamentSize = getTournamentSize();
		int pick = Mat.random(tournamentSize);
		for(int i = 1; i < tournamentSize; i ++){
			pick = Math.min(pick, Mat.random(tournamentSize));
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
	private TreeSTGP crossover(TreeSTGP p1, TreeSTGP p2) {
		TreeSTGP candidate, smaller = TreeSTGPCrossoverHandler.crossover(p1, p2);
		for(int i = 1; i < getTournamentSize(); i++) {
			candidate = TreeSTGPCrossoverHandler.crossover(p1, p2);
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
	private TreeSTGP mutation(TreeSTGP p) {
		TreeSTGP candidate, smaller = TreeSTGPMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
		for(int i = 1; i < getTournamentSize(); i++) {
			candidate = TreeSTGPMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
}