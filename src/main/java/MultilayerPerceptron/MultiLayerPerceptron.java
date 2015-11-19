package MultiLayerPerceptron;

import Model.Node;
import Model.Topology;
import Model.Weight;
import Util.Util;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class MultiLayerPerceptron extends Classifier {
    private Topology topology;
    private Instances dataset;
    private NominalToBinary nominalToBinaryFilter;

    public MultiLayerPerceptron()
    {
        topology = new Topology();
        nominalToBinaryFilter = new NominalToBinary();
    }

    MultiLayerPerceptron(Topology t)
    {
        topology = t;
        nominalToBinaryFilter = new NominalToBinary();
    }

    public Topology getTopology() {
        return topology;
    }

    public void setTopology(Topology topology) {
        this.topology = topology;
    }

    public Instances getDataset() {
        return dataset;
    }

    public void setDataset(Instances dataset) {
        this.dataset = dataset;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        dataset = new Instances(data);

        /* Delete all instances with missing value */
        for(int i=0; i<dataset.numAttributes(); i++)
        {
            dataset.deleteWithMissing(i);
        }

        /* filter dataset, convert nominal ke binary */
        nominalToBinaryFilter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, nominalToBinaryFilter);

        /* Update topologi dengan input neuron & output neuron
         * Jumlah input neuron = jumlah atribut dari data
         * Jumlah output neuron = jumlah kelas dari data
        */
        topology.addInputLayer(dataset.numAttributes() - 1);
        topology.addOutputLayer(dataset.numClasses());

        /* Connect all nodes in this topology */
        connectAllNodes();

        /* Training MultiLayerPerceptron berdasarkan dataset */
        /* Tom Mitchell, page 110/421 */

        for(int i=0; i<dataset.numInstances(); i++)
        {
            /* Propagate the input forward through the network */
            /* Input the instance x to the network */
            Instance instance = dataset.instance(i);
            topology.initInputNodes(instance);
            topology.initOutputNodes(instance);

            /* compute the output of every unit in the network */
            topology.sortWeight(false, true);

            /* Reset input node, init biased node */
            topology.resetNodesInput();

            /* Hitung input untuk setiap node */
            /* Propagate the input forward through the network */
            /* Input instance x to the network and compute the output of every unit in the network */
            System.out.println(topology.getWeights());
            Node currentNode = topology.getWeights().get(0).getNode2();
            for(Weight w : topology.getWeights())
            {
                w.getNode2().setInput(w.getNode2().getInput() + w.getWeight() * w.getNode1().getOutput());
                if(currentNode.getId() != w.getNode2().getId())
                {
                    currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
                    currentNode.setOutput(Node.siegmoid(currentNode.getInput()));
                    currentNode = w.getNode2();
                }
            }
            currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
            currentNode.setOutput(Node.siegmoid(currentNode.getInput()));

            /* Propagate the errors backward through the network */
            /* Calculate each output node error term */
            for(int j=0; j<topology.getLayers().get(topology.getLayers().size()-1); j++)
            {
                Node node = topology.getOutputNode(j);
                double error = node.getOutput() * (1 - node.getOutput()) * (node.getTarget() - node.getOutput());
                node.setError(error);
            }

            /* Calculate each hidden node error term */
            topology.sortWeight(true, false);
            topology.resetNodeError();

            /* jumlah Whk * error k */
            currentNode = topology.getWeights().get(0).getNode1();
            for(Weight w : topology.getWeights())
            {
                w.getNode1().setError(w.getNode1().getError() + w.getNode2().getError() * w.getWeight());
                if(currentNode.getId() != w.getNode1().getId())
                {
                    currentNode.setError(currentNode.getError() * currentNode.getOutput() * (1 - currentNode.getOutput()));
                    currentNode = w.getNode1();
                }
            }
            currentNode.setError(currentNode.getError() * currentNode.getOutput() * (1 - currentNode.getOutput()));
            System.out.println();
        }
    }

    /**
     * Makes all neurons in this topology fully connected
     */
    private void connectAllNodes()
    {
        topology.connectAllNodes();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    public static void main(String [] args) throws Exception {
        Instances dataset = Util.readARFF("simplified.weather.numeric.arff");
        Topology topology = new Topology();
        topology.addHiddenLayer(2);
        topology.setInitialWeight(0.1);
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(topology);
        mlp.buildClassifier(dataset);
    }
}
