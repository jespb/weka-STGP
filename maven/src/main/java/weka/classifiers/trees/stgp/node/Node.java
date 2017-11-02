package weka.classifiers.trees.stgp.node;

import java.io.Serializable;

import weka.classifiers.trees.stgp.util.Mat;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class Node implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	String v ;
	Node l ;
	Node r ;

	/**
	 * Basic constructor
	 * @param value
	 */
	public Node(String value){
		v = value;
	}

	/**
	 * Basic constructor
	 * @param left
	 * @param right
	 * @param op
	 */
	public Node(Node left, Node right, String op){
		l = left;
		r = right;
		v = op;
	}

	/**
	 * Constructor
	 * @param op available operators
	 * @param term available terminals
	 * @param t_rate probability of a node being terminal
	 * @param depth max depth from this node down
	 */
	public Node(String [] op, String [] term, double t_rate, int depth){
		if(depth <= 0 || Math.random() < t_rate){
			v = term[Mat.random(term.length)];
		}else{
			v = op[Mat.random(op.length)];
			l = new Node(op, term, t_rate, depth-1);
			r = new Node(op, term, t_rate, depth-1);
		}
	}

	/**
	 * Used the node to calculate
	 * @param vals
	 * @return
	 */
	public double calculate(double [] vals){
		if(isLeaf()){
			return vals[Integer.parseInt(v.substring(1))]; // o valor do nome eh "x_" onde _ eh um indice nos vals
		}else{
			double d = 0;
			switch(v){
			case "+":
				d = l.calculate(vals)+r.calculate(vals);
				break;
			case "-":
				d = l.calculate(vals)-r.calculate(vals);
				break;
			case "*":
				d = l.calculate(vals)*r.calculate(vals);
				break;
			case "/":
				double div = r.calculate(vals);
				d = l.calculate(vals)/(div != 0 ? div : 1);
				break;
			}
			return d;
		}
	}

	/**
	 * Returns the number of nodes on this object, he himself included
	 * @return
	 */
	public int getSize(){
		if (isLeaf())
			return 1;
		else
			return 1 + l.getSize() + r.getSize();
	}

	/**
	 * Clones a node
	 */
	public Node clone(){
		if(isLeaf()){
			return new Node(v);
		}else{
			return new Node(l.clone(), r.clone(), v);
		}
	}

	/**
	 * Returns the node under the String format
	 */
	public String toString(){
		if(isLeaf()){
			return v;
		}else{
			return "( " + l + " " + v + " " + r + " )";
		}
	}
	
	/**
	 * Returns true if the node is a leaf
	 * It's assumed that the node either had two children or none
	 * for a faster response
	 * @requires l==null <-> r==null
	 * @return
	 */
	private boolean isLeaf(){
		return l==null;// &&r==null;
	}
}