package Model;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

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
    private boolean useInitialWeight;
    /**
     * Jumlah iterasi pada saat training model
     */
    private double numIterations;
    /**
     * True if termination condition based on number of iterations
     * False if termination condition based on epoch error
     */
    private boolean useIteration;
    /**
     * Do training until epoch error under this value
     */
    private double epochErrorThreshold;
    /**
     * True if termination condition based on epoch error
     * False if termination condition based on number of iterations
     */
    private boolean useErrorThreshold;
    /**
     * MultiLayerPerceptron learning rate
     */
    private double learningRate;
    /**
     * MultiLayerPerceptron momentumRate
     */
    private double momentumRate;

    public Topology()
    {
        nodes = new ArrayList<>();
        weights = new ArrayList<>();
        layers = new ArrayList<>();
        initialWeight = 0.0;
        useInitialWeight = false;
        numIterations = 0;
        useIteration = false;
        epochErrorThreshold = 0.0;
        useErrorThreshold = false;
        learningRate = 0.0;
        momentumRate = 0.0;
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

    public double getInitialWeight() {
        return initialWeight;
    }

    public void setInitialWeight(double initialWeight) {
        this.initialWeight = initialWeight;
        useInitialWeight = true;
    }

    public boolean isUseInitialWeight() {
        return useInitialWeight;
    }

    public void setUseInitialWeight(boolean isInitialWeightSet) {
        this.useInitialWeight = isInitialWeightSet;
    }

    public double getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(double numIterations) {
        this.numIterations = numIterations;
    }

    public boolean isUseIteration() {
        return useIteration;
    }

    public void setUseIteration(boolean useIteration) {
        this.useIteration = useIteration;
    }

    public double getEpochErrorThreshold() {
        return epochErrorThreshold;
    }

    public void setEpochErrorThreshold(double epochErrorThreshold) {
        this.epochErrorThreshold = epochErrorThreshold;
    }

    public boolean isUseErrorThreshold() {
        return useErrorThreshold;
    }

    public void setUseErrorThreshold(boolean useErrorThreshold) {
        this.useErrorThreshold = useErrorThreshold;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getMomentumRate() {
        return momentumRate;
    }

    public void setMomentumRate(double momentumRate) {
        this.momentumRate = momentumRate;
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
     * Set the weight for all bias
     */
    private void setBiasWeight()
    {
        for(Node node : nodes)
        {
            node.setBiasWeight(initialWeight);
        }
    }

    /**
     * Set the bias value for all nodes in this topology
     * @param bias bias value for all nodes
     */
    public void setBiasValue(double bias)
    {
        for(Node n : nodes)
        {
            n.setBiasValue(bias);
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
        if(useInitialWeight)
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

    /**
     * Init the input node using the given data
     * @param data data for input to ANN
     */
    public void initInputNodes(Instance data)
    {
        for(int i=0; i<data.numAttributes()-1; i++)
        {
            nodes.get(i).setOutput(data.value(i));
        }
    }

    public void initOutputNodes(Instance data)
    {
        int classValue = (int) data.classValue();
        for(int i=0; i<data.numClasses(); i++)
        {
            Node n = getOutputNode(i);
            if(i == classValue)
            {
                n.setTarget(1);
            }
            else
            {
                n.setTarget(0);
            }
        }
    }

    /**
     * Sort weights based on node 1 or node 2, ascending or descending
     * @param useNode1 Sort based on node 1
     * @param ascending Sort ascending
     */
    public void sortWeight(final boolean useNode1, final boolean ascending)
    {
        Collections.sort(weights, new Comparator<Weight>() {
            @Override
            public int compare(Weight o1, Weight o2) {
                if(useNode1)
                {
                    if(ascending)
                    {
                        if(o1.getNode1().getId() > o2.getNode1().getId())
                        {
                            return 1;
                        }
                        else if (o1.getNode1().getId() == o2.getNode1().getId())
                        {
                            return 0;
                        }
                        else
                        {
                            return -1;
                        }
                    }
                    else // descending
                    {
                        if(o1.getNode1().getId() > o2.getNode1().getId())
                        {
                            return -1;
                        }
                        else if (o1.getNode1().getId() == o2.getNode1().getId())
                        {
                            return 0;
                        }
                        else
                        {
                            return 1;
                        }
                    }
                }
                else // based on node 2
                {
                    if(ascending)
                    {
                        if(o1.getNode2().getId() > o2.getNode2().getId())
                        {
                            return 1;
                        }
                        else if (o1.getNode2().getId() == o2.getNode2().getId())
                        {
                            return 0;
                        }
                        else
                        {
                            return -1;
                        }
                    }
                    else // descending
                    {
                        if(o1.getNode2().getId() > o2.getNode2().getId())
                        {
                            return -1;
                        }
                        else if (o1.getNode2().getId() == o2.getNode2().getId())
                        {
                            return 0;
                        }
                        else
                        {
                            return 1;
                        }
                    }
                }
            }
        });
    }

    /**
     * Get the n-th output node
     * @param n the index of output node, start from 0
     * @return
     */
    public Node getOutputNode(int n)
    {
        return nodes.get(nodes.size()-layers.get(layers.size()-1)+n);
    }

    /**
     * Reset all node's input to 0
     */
    public void resetNodeInput()
    {
        for(Node node: nodes)
        {
            node.setInput(0);
        }
    }
}
