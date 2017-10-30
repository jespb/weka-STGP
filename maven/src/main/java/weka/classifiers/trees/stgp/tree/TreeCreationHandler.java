package weka.classifiers.trees.stgp.tree;

public class TreeCreationHandler {
	/**
	 * Creates a new Tree
	 * @param op available operators
	 * @param term available terminals
	 * @param t_rate terminal rate
	 * @param depth max Depth
	 * @return
	 */
	public static Tree create(String [] op, String [] term, double t_rate, int depth) {
		return new Tree(op, term, t_rate, depth);
	}
}