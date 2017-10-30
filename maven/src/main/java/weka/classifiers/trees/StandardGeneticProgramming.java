package weka.classifiers.trees;

import java.util.Collections;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.stgp.forest.Forest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class StandardGeneticProgramming extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 1L;

	private Forest forest;

	private int populationSize = 200;
	private int maxGen = 40;
	private int maxDepth = 7;

	@Override
	/**
	 * Builds and trains the classifier
	 * @param data
	 * @throws Exception
	 */
	public void buildClassifier(Instances data) throws Exception {
		System.out.println("XXX: " + maxGen);
		int n_cols = data.numAttributes();
		int n_rows = data.numInstances();

		double [][] dados = new double [n_rows][n_cols];
		double [] target = new double [n_rows];

		for(int y = 0; y < n_rows; y++){
			Instance row = data.get(y);
			for(int x = 0; x < row.numAttributes(); x++){
				if(x < row.numAttributes() - 1)
					dados[y][x] = row.value(x);
				else
					target[y] = row.value(x);
			}
		}

		String [] op = "+ - * /".split(" ");

		String [] term = new String [n_cols];
		for(int i = 0; i < term.length; i++)
			term[i] = "x"+i;

		double train_perc = 0.7;

		forest = new Forest("",op, term, maxDepth,dados, target, populationSize, train_perc,"Ramped", maxGen);

		forest.train();
	}

	@Override
	/**
	 * Classifies an instance
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double classifyInstance(Instance instance) throws Exception{
		double [] dados = new double [instance.numAttributes()];
		for(int i = 0; i < dados.length; i++){
			dados[i] = instance.value(i);
		}
		return forest.predict(dados);
	}

	/**
	 * Gets options from this classifier.
	 * 
	 * @return the options for the current setup
	 */
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();

		result.add("-populationSize");
		result.add("" + populationSize);

		result.add("-initialMaxDepth");
		result.add("" + maxDepth);

		result.add("-maxGeneration");
		result.add("" + maxGen);

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Sets options
	 */
	public void setOptions(String[] options) throws Exception {
		String tmpStr;

		tmpStr = Utils.getOption("populationSize", options);
		if (tmpStr.length() != 0) {
			populationSize = Integer.parseInt(tmpStr);
		} else {
			populationSize = 200;
		}

		tmpStr = Utils.getOption("initialMaxDepth", options);
		if (tmpStr.length() != 0) {
			maxDepth = Integer.parseInt(tmpStr);
		} else {
			maxDepth = 7;
		}

		tmpStr = Utils.getOption("maxGeneration", options);
		if (tmpStr.length() != 0) {
			maxGen = Integer.parseInt(tmpStr);
		} else {
			maxGen = 40;
		}
	}

	public String toString(){
		return forest.toString();
	}

}