package MultiLayerPerceptron;

import Model.Node;
import Model.Topology;
import Model.Weight;
import Util.Util;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.Serializable;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class MultiLayerPerceptron extends Classifier implements Serializable {
    private Topology topology;
    private Instances dataset;
    private NominalToBinary nominalToBinaryFilter;
    private Normalize normalizeFilter;

    MultiLayerPerceptron(Topology t)
    {
        topology = new Topology(t);
        nominalToBinaryFilter = new NominalToBinary();
        normalizeFilter = new Normalize();
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

        /* filter normalization for numeric input */
        normalizeFilter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, normalizeFilter);

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
        int iterations = 0; // number of iteration
        while(true)
        {
            for (int i = 0; i < dataset.numInstances(); i++)
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
                Node currentNode = topology.getWeights().get(0).getNode2();
                for (Weight w : topology.getWeights()) {
                    w.getNode2().setInput(w.getNode2().getInput() + w.getWeight() * w.getNode1().getOutput());
                    if (currentNode.getId() != w.getNode2().getId()) {
                        currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
                        currentNode.setOutput(Node.siegmoid(currentNode.getInput()));
                        currentNode = w.getNode2();
                    }
                }
                currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
                currentNode.setOutput(Node.siegmoid(currentNode.getInput()));

                /* Propagate the errors backward through the network */
                /* Calculate each output node error term */
                for (int j = 0; j < topology.getLayers().get(topology.getLayers().size() - 1); j++) {
                    Node node = topology.getOutputNode(j);
                    double error = node.getOutput() * (1 - node.getOutput()) * (node.getTarget() - node.getOutput());
                    node.setError(error);
                }

                /* Calculate each hidden node error term */
                topology.sortWeight(true, false);
                topology.resetNodeError();

                /* jumlah Whk * error k */
                currentNode = topology.getWeights().get(0).getNode1();
                for (Weight w : topology.getWeights()) {
                    w.getNode1().setError(w.getNode1().getError() + w.getNode2().getError() * w.getWeight());
                    if (currentNode.getId() != w.getNode1().getId()) {
                        currentNode.setError(currentNode.getError() * currentNode.getOutput() * (1 - currentNode.getOutput()));
                        currentNode = w.getNode1();
                    }
                }
                currentNode.setError(currentNode.getError() * currentNode.getOutput() * (1 - currentNode.getOutput()));

                /* Update each network weight */
                topology.sortWeight(false, true);

                /* Update weight between 2 nodes */
                for (Weight w : topology.getWeights()) {
                    double deltaWeight = topology.getLearningRate() * w.getNode2().getError() * w.getNode1().getOutput();
                    deltaWeight = deltaWeight + topology.getMomentumRate() * w.getPreviousDeltaWeight();
                    w.setWeight(w.getWeight() + deltaWeight);
                    w.setPreviousDeltaWeight(deltaWeight);
                }

                /* Update bias weight */
                for (Node n : topology.getNodes()) {
                    double deltaWeight = topology.getLearningRate() * n.getBiasValue() * n.getError();
                    deltaWeight = deltaWeight + topology.getMomentumRate() * n.getPreviousDeltaWeight();
                    n.setBiasWeight(n.getBiasWeight() + deltaWeight);
                    n.setPreviousDeltaWeight(deltaWeight);
                }
            }

            /* Check if termination condition is satisfied */
            if(topology.isUseIteration())
            {
                iterations++;
                if(iterations >= topology.getNumIterations())
                {
                    break;
                }
            }

            if(topology.isUseErrorThreshold())
            {
                double epochError = 0;
                for(int i=0; i<dataset.numInstances(); i++)
                {
                    Instance instance = dataset.instance(i);
                    Node outputNode = topology.getOutputNode((int)classifyInstance(instance));
                    epochError = epochError + Math.pow(outputNode.getTarget() - outputNode.getOutput(),2);
                }
                epochError = epochError / 2;

                if (epochError <= topology.getEpochErrorThreshold())
                {
                    break;
                }
            }
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
    public double classifyInstance(Instance data) throws Exception {
        Instance instance = new Instance(data);

        /* Filter instance */
        nominalToBinaryFilter.input(instance);
        instance = nominalToBinaryFilter.output();

        normalizeFilter.input(instance);
        instance = normalizeFilter.output();

        topology.initInputNodes(instance);
        topology.initOutputNodes(instance);

        /* compute the output of every unit in the network */
        topology.sortWeight(false, true);

        /* Reset input node, init biased node */
        topology.resetNodesInput();

        /* Hitung input untuk setiap node */
        /* Propagate the input forward through the network */
        /* Input instance x to the network and compute the output of every unit in the network */
        Node currentNode = topology.getWeights().get(0).getNode2();
        for (Weight w : topology.getWeights()) {
            w.getNode2().setInput(w.getNode2().getInput() + w.getWeight() * w.getNode1().getOutput());
            if (currentNode.getId() != w.getNode2().getId()) {
                currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
                currentNode.setOutput(Node.siegmoid(currentNode.getInput()));
                currentNode = w.getNode2();
            }
        }
        currentNode.setInput(currentNode.getInput() + (currentNode.getBiasWeight() * currentNode.getBiasValue()));
        currentNode.setOutput(Node.siegmoid(currentNode.getInput()));

        double maxOutput = 0;
        double maxIndex = 0;
        for (int i = 0; i < topology.getLayers().get(topology.getLayers().size() - 1); i++)
        {
            Node outputNode = topology.getOutputNode(i);
            if(outputNode.getOutput() > maxOutput)
            {
                maxOutput = outputNode.getOutput();
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();

        topology.sortWeight(true, true);
        topology.sortWeight(false, true);

        for (Weight w : topology.getWeights()){
            result.append("Weight[").append(w.getNode1().getId()).append("][").append(w.getNode2().getId()).append("]: ").append(w.getWeight()).append("\n");
        }

        for (int i=topology.getLayers().get(0); i<topology.getNodes().size(); i++) {
            Node n = topology.getNodes().get(i);
            result.append("Bias Node ").append(n.getId()).append(": ").append(n.getBiasWeight()).append("\n");
        }

        return result.toString();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double result[] = new double[instance.numClasses()];
        double classValue = classifyInstance(instance);
        for(int i=0; i<instance.numClasses(); i++){
            if((int)classValue == i){
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        return result;
    }

    public static void main(String [] args) throws Exception {
        String datasetName = "car.arff";
        Instances dataset = Util.readARFF(datasetName);

        Topology topology = new Topology();
//        topology.addHiddenLayer(5);
        topology.addHiddenLayer(5);
        topology.setLearningRate(0.1);
        topology.setMomentumRate(0.1);
        topology.setNumIterations(500);
        topology.setEpochErrorThreshold(0.0);

        Classifier multiLayerPerceptron = new MultiLayerPerceptron(topology);
        multiLayerPerceptron.buildClassifier(dataset);

//        Evaluation eval = Util.evaluateModel(multiLayerPerceptron, dataset);
        Evaluation eval = Util.crossValidationTest(dataset, new MultiLayerPerceptron(topology));
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toSummaryString());
    }
}
