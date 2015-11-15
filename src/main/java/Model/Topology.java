package Model;

import java.util.ArrayList;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Topology {
    private ArrayList<Node> nodes;
    private ArrayList<Weight> weights;

    public Topology()
    {
        nodes = new ArrayList<>();
        weights = new ArrayList<>();
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
