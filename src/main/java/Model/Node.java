package Model;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Node {
    /**
     * Error for this node
     */
    private double error;
    /**
     * Target output for this node
     */
    private double target;
    /**
     * Output value from this node
     */
    private double output;
    /**
     * Bias weight for this node
     * Bias value always 1
     */
    private double bias;

    /**
     * Compute the siegmoid value from x
     * @param x input value to neuron
     * @return siegmoid(x)
     */
    public static double siegmoid(double x)
    {
        return 1 / (1 + (Math.exp(-x)));
    }

    /**
     * Compute the sign value from x
     * @param x input value to neuron
     * @return sign(x)
     */
    public static int sign(double x)
    {
        if(x > 0)
        {
            return 1;
        }
        else
        {
            return -1;
        }
    }

    /**
     * Test siegmoid(x) and sign(x) function
     */
    public static void testOutputFunction()
    {
        double x = 5;
        System.out.printf("Siegmoid(%d): %f\n", (int)x, Node.siegmoid(x));
        System.out.printf("Sign(%d): %d\n", (int)x, Node.sign(x));
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getTarget() {
        return target;
    }

    public void setTarget(double target) {
        this.target = target;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public static void main (String [] args)
    {
        Node.testOutputFunction();
    }
}
