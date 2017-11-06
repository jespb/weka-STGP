package weka.classifiers.trees.stgp.client;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.trees.stgp.node.Node;
import weka.classifiers.trees.stgp.population.Population;
import weka.classifiers.trees.stgp.util.Arrays;
import weka.classifiers.trees.stgp.util.Data;
import weka.classifiers.trees.stgp.util.Files;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class ClientWekaSim {

	private static int file = 0; // ST, GS

	private static String xDataInputFilename = "Brazil_x.txt bioavailability_x.txt".split(" ")[file];
	private static String yDataInputFilename = "Brazil_y.txt bioavailability_y.txt".split(" ")[file];
	private static String resultOutputFilename = "fitovertime.csv";
	private static String treeType = "Ramped";

	private static String [] operations = "+ - * /".split(" ");
	private static String [] terminals = null;

	private static double trainFraction = 0.70;
	private static double tournamentFraction = 0.02;
	private static double elitismFraction = 0.01;

	private static int numberOfGenerations = 50;
	private static int numberOfRuns = 1;
	private static int populationSize = 250;
	private static int maxDepth = 7;

	private static boolean shuffleDataset = true;

	private static double [][] data = null;
	private static double [] target = null;


	// Variables
	public static double [][] results = new double [numberOfGenerations][3];
	static Population f = null;

	/**
	 * main
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		treatArgs(args);
		init();

		long time = System.currentTimeMillis();
		for(int run = 0 ; run < numberOfRuns; run++){
			run(run);
		}
		System.out.println((System.currentTimeMillis() - time) + "ms");

		BufferedWriter out = new BufferedWriter(new FileWriter(resultOutputFilename+".tmp"));
		out.write("Treino;Teste\n");
		for(int i = 0; i < results.length; i++){
			if(results[i][2] !=0)
				out.write(results[i][0]/results[i][2] + ";" + results[i][1]/results[i][2] + "\n");
		}
		out.close();
		Files.fixCSV(resultOutputFilename);
	}

	/**
	 * Prepara o cliente para a sua execucao
	 * @throws IOException
	 */
	private static void init() throws IOException{
		data = Data.readData(xDataInputFilename);
		target = Data.readTarget(yDataInputFilename);
	}

	/**
	 * Executa uma simulacao
	 * @param run
	 * @throws IOException
	 */
	private static void run(int run) throws IOException{
		System.out.println("Run " + run + ":");

		if(shuffleDataset)Arrays.shuffle(data, target);

		setTerminals(data);

		double [][] train = new double [(int) (data.length*trainFraction)][data[0].length];
		double [][] test = new double [data.length - train.length][data[0].length];

		for(int i = 0; i < data.length; i++){
			if( i < train.length)
				train[i] = data[i];
			else
				test[i - train.length] = data[i];
		}

		setPopulation();

		f.train();


		// Este bloco está a certificarse que as previsoes sao consistentes com o treino
		double acc = 0;
		int hit = 0;
		double prediction = 0;
		for(int i = 0; i < test.length; i++){
			prediction= f.predict(test[i]) ; Math.pow( f.predict(test[i]) - target[train.length + i] ,2);
			acc += Math.pow( prediction - target[train.length + i] ,2);
			if((target[train.length + i] < 0.5 && prediction<0.5)||
					(target[train.length + i] >= 0.5 && prediction>=0.5)) hit++;
			if((i+1)%500 ==0)
				System.out.println((i+1) + "/" + test.length);
		}
		acc /= 1.0 * test.length;
		acc = Math.sqrt(acc);

		
		System.out.println("test binary classification hits: " + hit +" out of " + test.length + "( " + 100.0*hit/test.length + "% )");
		System.out.println("test RMSE calculated: " + acc);

		acc = 0;
		hit = 0;

		for(int i = 0; i < train.length; i++){
			prediction = f.predict(train[i]);Math.pow( - target[i],2);
			acc+= Math.pow(prediction - target[i],2);
			if((target[i] < 0.5 && prediction<0.5)||
					(target[i] >= 0.5 && prediction>=0.5)) hit++;
			if((i+1)%500 ==0)
				System.out.println((i+1) + "/" + train.length);
		}
		acc /= 1.0 * train.length;
		acc = Math.sqrt(acc);

		System.out.println("train binary classification hits: " + hit +" out of " + train.length + "( " + 100.0*hit/train.length + "% )");
		System.out.println("train RMSE calculated: " + acc);

		
		System.out.println(f);
	}

	/**
	 * Trata dos argumentos fornecidos
	 * @param args
	 */
	private static void treatArgs(String [] args){
		for(int i = 0; i < args.length; i++){
			String [] split = args[i].split(":");
			switch(split[0]){
			case "depth":
				maxDepth = Integer.parseInt(split[1]);
				break;
			case "maxgen":
				numberOfGenerations = Integer.parseInt(split[1]);
				break;
			case "popsize":
				populationSize = Integer.parseInt(split[1]);
				break;
			}
		}
	}

	/**
	 * Define o valor dos terminais
	 * @param data
	 */
	private static void setTerminals(double [][] data){
		terminals = new String [data[0].length];
		for(int i = 0; i < terminals.length; i++)
			terminals[i] = "x"+i;
	}

	/**
	 * Cria uma nova floresta
	 * @throws IOException
	 */
	private static void setPopulation() throws IOException{
		f = new Population("", operations, 
				terminals, maxDepth, data, target, 
				populationSize,trainFraction, treeType,numberOfGenerations,
				tournamentFraction, elitismFraction);
	}
}