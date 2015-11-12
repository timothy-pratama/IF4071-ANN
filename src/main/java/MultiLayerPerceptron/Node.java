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

    /**
     * Nilai target untuk node ini
     */
    private double target;

    public Node()
    {
        output = 0.0;
        error = 0.0;
        bias = 0.0;
        target = 0.0;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getTarget() {
        return target;
    }

    public void setTarget(double target) {
        this.target = target;
    }

    public void computeOutputError()
    {

    }

    public void computeHiddenUnitError()
    {

    }

    public static double siegmoid(double input)
    {
        return (double)1/(double)(1+Math.exp(-input));
    }

    public static void main(String [] args)
    {
        Node node = new Node();
        node.setOutput(Node.siegmoid(-0.05625));
    }
}
