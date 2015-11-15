package Model;

import java.util.ArrayList;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Topology {
    /**
     * Semua node yang ada di topologi ini
     */
    private ArrayList<Node> nodes;
    /**
     * Semua weight dalam topologi ini
     */
    private ArrayList<Weight> weights;
    /**
     * jumlah neuron pada setiap layer
     * layer pertama  -> input layer
     * layer n        -> hidden layer
     * layer terakhir -> output layer
     */
    private ArrayList<Integer> layers;

    public Topology()
    {
        nodes = new ArrayList<>();
        weights = new ArrayList<>();
        layers = new ArrayList<>();
    }

    public ArrayList<Node> getNodes() {
        return nodes;
    }

    public void setNodes(ArrayList<Node> nodes) {
        this.nodes = nodes;
    }

    public ArrayList<Weight> getWeights() {
        return weights;
    }

    public void setWeights(ArrayList<Weight> weights) {
        this.weights = weights;
    }

    public ArrayList<Integer> getLayers() {
        return layers;
    }

    public void setLayers(ArrayList<Integer> layers) {
        this.layers = layers;
    }

    /**
     * Add a new hidden layer with n neurons
     * @param n number of neurons
     */
    public void addHiddenLayer(int n)
    {
        layers.add(n);
    }

    /**
     * Add a new input layer with n input attributes
     * @param n number of input attributes
     */
    public void addInputLayer(int n)
    {
        layers.add(0, n);
    }

    /**
     * add a new output layer with n classes
     * @param n number of output class
     */
    public void addOutputLayer(int n)
    {
        layers.add(layers.size(), n);
    }

    public void removeLayer(int index)
    {
        layers.remove(index);
    }

    /**
     * Set the weight for all nodes and bias
     * @param weight the new weight
     */
    public void setWeight(double weight)
    {
        for(Node node : nodes)
        {
            node.setBias(weight);
        }
        for(Weight w : weights)
        {
            w.setWeight(weight);
        }
    }
}
