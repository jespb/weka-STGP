package weka.classifiers.trees.stgp.tree;

import weka.classifiers.trees.stgp.node.Node;
import weka.classifiers.trees.stgp.node.NodeHandler;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class TreeMutationHandler {
	/**
	 * Mutates a TreeSTGP by selection a random node and replacing it with a new TreeSTGP
	 * This method fails if a randomly generated number if lower than the mutation odd
	 * @param t The tree to mutate
	 * @param op available operators
	 * @param term available terminals
	 * @param t_rate Odds of a node being terminal
	 * @param depth max depth of the tree
	 * @return
	 */
	public static Tree mutation(Tree t, String[] op, String[] term,double t_rate, int depth){		
		Node p1 = t.getHead().clone();
		
		Node r1 = NodeHandler.randomNode(p1);
		
		NodeHandler.redirect(r1, new Node(op,term,t_rate,depth));
		
		return new Tree(p1);
	}
}