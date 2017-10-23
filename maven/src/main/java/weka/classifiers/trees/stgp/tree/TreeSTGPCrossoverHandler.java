package weka.classifiers.trees.stgp.tree;

import weka.classifiers.trees.stgp.node.Node;
import weka.classifiers.trees.stgp.node.NodeHandler;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class TreeSTGPCrossoverHandler {	
	/**
	 * Executes the crossover of a TreeSTGP by swaping a random node with 
	 * another random node from other TreeSTGP
	 * This method fails if a randomly generated number is lower than the 
	 * average of the Trees crossover odds
	 * @param parent1
	 * @param parent2
	 * @return
	 */
	public static TreeSTGP crossover(TreeSTGP parent1, TreeSTGP parent2){
		Node p1 = parent1.getHead().clone();
		Node p2 = parent2.getHead();

		Node r1 = NodeHandler.randomNode(p1);
		Node r2 = NodeHandler.randomNode(p2);

		NodeHandler.redirect(r1,r2);

		return new TreeSTGP(p1);
	}
}