package PerceptronTrainingRule;

import Model.Node;
import Model.Topology;
import Model.Weight;
import Util.Util;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.Serializable;

/**
 * Created by timothy.pratama on 07-Nov-15.
 */
public class PerceptronTrainingRule extends Classifier implements Serializable{
    private Topology topology;
    private Instances dataset;
    private NominalToBinary nominalToBinaryFilter = new NominalToBinary();
    private Normalize normalizeFilter = new Normalize();

    public PerceptronTrainingRule(){
        topology = new Topology();
    }

    public PerceptronTrainingRule(Topology t){
        topology = t;
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
        getCapabilities().testWithFail(data);
        dataset = new Instances(data);
        for(int i=0; i<dataset.numAttributes(); i++)
        {
            dataset.deleteWithMissing(i);
        }
        nominalToBinaryFilter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, nominalToBinaryFilter);
        normalizeFilter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, normalizeFilter);
        topology.addInputLayer(dataset.numAttributes() - 1);
        topology.addOutputLayer(1);

        topology.connectAllNodes();

        int epochError;
        int tries = 0;
        double previousdelta = 0.0;
        Node outputNode = topology.getOutputNode(0);
        outputNode.setBiasWeight(0.0);
        double treshold = topology.isUseErrorThreshold()?topology.getEpochErrorThreshold():0.0;
        do {
            epochError = 0;
            for (int i = 0; i < dataset.numInstances(); i++) {
                Instance instance = dataset.instance(i);
                int target = ((int) instance.classValue())==0?-1:1;
                topology.initInputNodes(instance);
                
                //topology.sortWeight(false, true);
                topology.resetNodesInput();

                for (int j = 0; j < topology.getWeights().size(); j++) {
                    Weight weight = topology.getWeights().get(j);
                    weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                }
                outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                int output = Node.sign(outputNode.getInput());
                for (int j = 0; j < topology.getWeights().size(); j++) {
                    Weight weight = topology.getWeights().get(j);
                    double delta = (topology.getLearningRate() * (target - output) * weight.getNode1().getOutput()) +
                            topology.getMomentumRate()* weight.getPreviousDeltaWeight();
                    weight.setPreviousDeltaWeight(delta);
                    weight.setWeight(weight.getWeight() + delta);
                }
                double biasWeight = outputNode.getBiasWeight();
                double delta = (topology.getLearningRate() * (target - output) * outputNode.getBiasValue()) +
                                topology.getMomentumRate()* outputNode.getPreviousDeltaWeight();
                outputNode.setPreviousDeltaWeight(delta);
                outputNode.setBiasWeight(biasWeight + delta);
                topology.resetNodesInput();
                for (int j = 0; j < topology.getWeights().size(); j++) {
                    Weight weight = topology.getWeights().get(j);
                    weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                }
                outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                output = Node.sign(outputNode.getInput());
                int squaredError = (output-target)*(output-target);
                epochError += squaredError;
            }
            epochError *= 0.5;
            tries++;
        }
        while((epochError > treshold) && (!topology.isUseIteration() || (tries < topology.getNumIterations())));

    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        Instance instance = new Instance(data);
        nominalToBinaryFilter.input(instance);
        instance = nominalToBinaryFilter.output();
        normalizeFilter.input(instance);
        instance = normalizeFilter.output();

        topology.initInputNodes(instance);
        Node outputNode = topology.getOutputNode(0);
        topology.resetNodesInput();

        for (int j = 0; j < topology.getWeights().size(); j++) {
            Weight weight = topology.getWeights().get(j);
            weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
        }
        outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
        int output = (Node.sign(outputNode.getInput())==-1)?0:1;
        return (double)output;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.BINARY_CLASS);

        return result;
    }

    public static void main(String [] args) throws Exception {
        Instances dataset = Util.readARFF("weather.nominal.arff");
        Topology topology = new Topology();
        topology.setLearningRate(0.1);
        topology.setInitialWeight(0.0);
        topology.setMomentumRate(0.0);
        topology.setNumIterations(500);
        Classifier ptr = new PerceptronTrainingRule(topology);
        ptr.buildClassifier(dataset);
        Evaluation eval = Util.evaluateModel(ptr, dataset);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
}
