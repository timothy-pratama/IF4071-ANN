package Model;

import java.util.Random;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Weight {
    /**
     * The first node
     */
    private Node node1;
    /**
     * The second node
     */
    private Node node2;
    /**
     * The weight for this node
     */
    private double weight;

    public Weight()
    {
        node1 = new Node();
        node2 = new Node();
        randomWeight();
    }

    public Weight(Node n1, Node n2, double weight)
    {
        node1 = n1;
        node2 = n2;
        this.weight = weight;
    }

    /**
     * Assign random value between -0.1 and 0.1 for the weight
     */
    private void randomWeight()
    {
        double rangeMin = -0.1;
        double rangeMax = 0.1;
        Random r = new Random();

        weight = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
    }

    public Node getNode1() {
        return node1;
    }

    public void setNode1(Node node1) {
        this.node1 = node1;
    }

    public Node getNode2() {
        return node2;
    }

    public void setNode2(Node node2) {
        this.node2 = node2;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("weight[").append(node1.getId()).append("][").append(node2.getId()).append("]: ").append(weight).append("\n");
        return sb.toString();
    }
}
