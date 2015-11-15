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
    /**
     * Initial weight for all nodes and bias
     */
    private double initialWeight;
    /**
     * True if initial weight is given, false if initial weight is not given
     */
    private boolean isInitialWeightSet;

    public Topology()
    {
        nodes = new ArrayList<>();
        weights = new ArrayList<>();
        layers = new ArrayList<>();
        initialWeight = 0;
        isInitialWeightSet = false;
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
     * Set the initial weight for all nodes and bias
     * @param weight the weight for all nodes and bias
     */
    public void setInitialWeight(double weight)
    {
        initialWeight = weight;
        isInitialWeightSet = true;
    }

    /**
     * Set the weight for all bias
     */
    private void setBiasWeight()
    {
        for(Node node : nodes)
        {
            node.setBias(initialWeight);
        }
    }

    /**
     * Connect all nodes in this topology
     */
    public void connectAllNodes()
    {
        createNodes();
        createWeights();
    }


    /**
     * Create nodes according to number of nodes in input layer, hidden layer, and output layer
     */
    private void createNodes()
    {
        int layerSize = 0;
        int id = 0;
        for(int i=0; i<layers.size(); i++)
        {
            layerSize = layers.get(i);
            for(int j=0; j<layerSize; j++)
            {
                nodes.add(new Node(id));
                id++;
            }
        }
    }

    /**
     * Create weights between nodes.
     */
    private void createWeights()
    {
        int currentLayerSize = 0;
        int nextLayerSize = 0;
        int baseID = 0;
        if(isInitialWeightSet)
        {
            setBiasWeight();
            for(int i=0; i<layers.size()-1; i++)
            {
                currentLayerSize = layers.get(i);
                nextLayerSize = layers.get(i+1);
                for(int j=0; j<currentLayerSize; j++)
                {
                    for(int k=0; k<nextLayerSize; k++)
                    {
                        weights.add(new Weight(nodes.get(baseID+j), nodes.get(baseID + currentLayerSize + k), initialWeight));
                    }
                }
                baseID += currentLayerSize;
            }
        }
        else // initial weight is not set
        {
            for(int i=0; i<layers.size()-1; i++)
            {
                currentLayerSize = layers.get(i);
                nextLayerSize = layers.get(i+1);
                for(int j=0; j<currentLayerSize; j++)
                {
                    for(int k=0; k<nextLayerSize; k++)
                    {
                        weights.add(new Weight(nodes.get(baseID+j), nodes.get(baseID + currentLayerSize + k)));
                    }
                }
                baseID += currentLayerSize;
            }
        }
    }
}
