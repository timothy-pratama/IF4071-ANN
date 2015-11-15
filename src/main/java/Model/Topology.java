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
