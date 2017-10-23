package weka.classifiers.trees.stgp.util;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class Arrays {
	/**
	 * Shuffle two arrays in the same way
	 * @param data
	 * @param target
	 */
	public static void shuffle(double[][] data, double[] target) {
		int n = data.length;
		for (int i = 0; i < data.length; i++) {
			// Get a random index of the array past i.
			int random = i + Mat.random(n-i);
			// Swap the random element with the present element.
			double[] randomElement = data[random];
			data[random] = data[i];
			data[i] = randomElement;

			double randomEl = target[random];
			target[random] = target[i];
			target[i] = randomEl;
		}
	}

	/**
	 * Uses merge sort to sort both arrays by the values in a
	 * @param o object array
	 * @param a array with values
	 */
	public static void mergeSortBy(Object[]o, double[]a){
		double [] b  = a.clone();
		Object [] bo = o.clone();
		topDownSplitMerge(b,bo,0,a.length,a,o);
		topDownMerge(a,o, 0, a.length/2, a.length, b,bo);
	}

	/**
	 * Sub-method of mergeSortBy
	 * @param b
	 * @param bo
	 * @param min
	 * @param max
	 * @param a
	 * @param o
	 */
	private static void topDownSplitMerge(double[] b, Object [] bo, int min, int max, double[] a, Object [] o) {
		if(max-min < 2)
			return;
		int mean = (max+min)/2;
		
		topDownSplitMerge(a,o, min, mean, b,bo);
		topDownSplitMerge(a,o, mean, max, b,bo);
		topDownMerge(b,bo, min, mean, max, a,o);
	}
	
	/**
	 * Sub-method of mergeSortBy
	 * @param b
	 * @param bo
	 * @param min
	 * @param mean
	 * @param max
	 * @param a
	 * @param o
	 */
	private static void topDownMerge(double[] b, Object [] bo, int min, int mean, int max, double[] a, Object [] o) {
		int i = min, j = mean;
		for(int k = min; k < max; k++){
			if(i<mean && (j >= max || a[i] <= a[j])){
				b[k] = a[i];
				bo[k] = o[i];
				i++;
			}else{
				b[k] = a[j];
				bo[k] = o[j];
				j++;
			}
		}
	}
}