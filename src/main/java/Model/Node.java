package Model;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Node implements Serializable{
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
     * Input for this node
     */
    private double input;
    /**
     * Bias Weight for this node
     */
    private double biasWeight;
    /**
     * Bias value for this node
     * Default value = 1
     */
    private double biasValue;
    /**
     * This node's ID
     */
    private int id;
    /**
     * This node previous delta weight. Used for calculation using momentum
     */
    private double previousDeltaWeight;

    public Node()
    {
        error = 0;
        target = 0;
        output = 0;
        input = 0;
        id = 0;
        randomBias();
        biasValue = 1;
        previousDeltaWeight = 0;
    }

    public Node(int id)
    {
        error = 0;
        target = 0;
        output = 0;
        input = 0;
        this.id = id;
        randomBias();
        biasValue = 1;
        previousDeltaWeight = 0;
    }

    /**
     * Random this node biasWeight weight
     */
    private void randomBias()
    {
        double rangeMin = -0.1;
        double rangeMax = 0.1;
        Random r = new Random();

        biasWeight = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
    }

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

    public double getBiasWeight() {
        return biasWeight;
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public double getBiasValue() {
        return biasValue;
    }

    public void setBiasValue(double biasValue) {
        this.biasValue = biasValue;
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public double getPreviousDeltaWeight() {
        return previousDeltaWeight;
    }

    public void setPreviousDeltaWeight(double previousDeltaWeight) {
        this.previousDeltaWeight = previousDeltaWeight;
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Node ").append(id).append("\n");
        sb.append("biasWeight: ").append(biasWeight).append("\n");
        sb.append("error: ").append(error).append("\n");
        sb.append("target: ").append(target).append("\n");
        sb.append("output: ").append(output).append("\n");
        sb.append("input: ").append(input).append("\n");
        return sb.toString();
    }

    public static void main (String [] args)
    {
        Node.testOutputFunction();
    }
}
