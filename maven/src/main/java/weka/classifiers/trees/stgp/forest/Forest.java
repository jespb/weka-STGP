package weka.classifiers.trees.stgp.forest;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public interface Forest extends Serializable {
	
	/**
	 * Trains the forest
	 * @throws IOException 
	 */
	public ArrayList<Double>[] train() throws IOException;
	
	/**
	 * Predicts a result given a data rown
	 * @param data data row
	 * @requires isTrained()
	 */
	public double predict(double [] data);
	
	/**
	 * Change the value of the percentage of the population on a tournament
	 * @param d
	 */
	public void setTournamentFraction(double d);
	
	/**
	 * Change the percentage of individuals on the population selected through elitism
	 * @param d
	 */
	public void setElitismFraction(double d);
}
