package weka.classifiers.trees.stgp.tree;

import java.io.Serializable;

import weka.classifiers.trees.stgp.node.Node;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Tree implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private Node head;

	/**
	 * Constructor
	 * @param head
	 */
	public Tree(Node head){
		this.head = head;
	}

	/**
	 * Constructor
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param depth
	 */
	public Tree(String [] op, String [] term, double t_rate, int depth){
		head = new Node(op, term, t_rate,depth);
	}

	/**
	 * Returns the TreeSTGP head
	 */
	public Node getHead(){
		return head;
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