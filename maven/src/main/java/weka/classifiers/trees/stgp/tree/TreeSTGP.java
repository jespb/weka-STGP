package weka.classifiers.trees.stgp.tree;

import java.io.Serializable;

import weka.classifiers.trees.stgp.node.Node;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class TreeSTGP implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static int trainSize;
	private Node head;

	/**
	 * Constructor
	 * @param name
	 * @param or
	 * @param head
	 * @param co crossover odds
	 * @param mo mutation odds
	 */
	public TreeSTGP(Node head){
		this.head = head;
	}

	/**
	 * Constructor
	 * @param name
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param depth
	 */
	public TreeSTGP(String [] op, String [] term, double t_rate, int depth){
		head = new Node(op, term, t_rate,0, depth);
	}

	/**
	 * Returns the TreeSTGP head
	 */
	public Node getHead(){
		return head;
	}

	/**
	 * Returns the train rmse
	 * @param data data
	 * @param target target values
	 * @param trainPercentage train percentage
	 * @return 
	 */
	public double getTrainRMSE(double[][] data, double[] target, double trainPercentage) {
		double sum = 0;
		//int trainSize = (int)(target.length * trainPercentage);

		for(int i = 0; i < trainSize;i++){
			sum += Math.pow(head.calculate(data[i])-target[i], 2);
		}

		sum /= trainSize;

		return Math.sqrt(sum);
	}

	/**
	 * Returns the test rmse
	 * @param data data
	 * @param target target values
	 * @param trainPercentage train percentage
	 * @return 
	 */
	public double getTestRMSE(double[][] data, double[] target, double trainPercentage) {
		double sum = 0;
		//int trainSize = (int)(target.length * trainPercentage);

		for(int i = trainSize; i < target.length ;i++){
			sum += Math.pow(head.calculate(data[i])-target[i], 2);
		}
		sum /= target.length - trainSize;

		return Math.sqrt(sum);
	}
	
	/**
	 * Returns the TreeSTGP under it's String format
	 */
	public String toString(){
		return head.toString();
	}
	
	/**
	 * Returns the number of nodes inside the tree
	 */
	public int size() {
		return head.size();
	}
}