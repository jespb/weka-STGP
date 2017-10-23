package weka.classifiers.trees.stgp.tree;

import weka.classifiers.trees.stgp.node.Node;
import weka.classifiers.trees.stgp.node.NodeHandler;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class TreeSTGPMutationHandler {
	/**
	 * Mutates a TreeSTGP by selection a random node and replacing it with a new TreeSTGP
	 * This method fails if a randomly generated number if lower than the mutation odd
	 * @param string
	 * @param treeSTGP
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param max_depth
	 * @return
	 */
	public static TreeSTGP mutation(TreeSTGP original, String[] op, String[] term, 
			double t_rate, int max_depth, double [][] data, double [] target, double train_p) {		
		Node p1 = original.getHead().clone();

		Node r1 = NodeHandler.randomNode(p1);

		NodeHandler.redirect(r1, new Node(op,term,t_rate,0,max_depth));

		return new TreeSTGP(p1);
	}
}