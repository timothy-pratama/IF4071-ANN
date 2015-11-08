package MultiLayerPerceptron;

/**
 * Created by timothy.pratama on 08-Nov-15.
 */
public class Node {
    /**
     * Nilai output dari node ini
     */
    private double output;
    /**
     * Nilai error untuk node ini
     */
    private double error;

    /**
     * Nilai bobot bias untuk node ini
     */
    private double bias;

    public Node()
    {
        output = 0.0;
        error = 0.0;
        bias = 0.0;
    }

    public double getOutput() {
        return output;
    }

    public double getError() {
        return error;
    }

    public void computeOutputError()
    {

    }

    public void computeHiddenUnitError()
    {

    }

    public void computeOutput(double input)
    {
        output = (double)1/(double)(1+Math.exp(-(input+bias)));
    }

    public static void main(String [] args)
    {
        Node node = new Node();
        node.computeOutput(-0.05625);
    }
}
