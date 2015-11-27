package DeltaTrainingRule;

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
 * Created by kevin.huang on 07-Nov-15.
 */
public class DeltaTrainingRule extends Classifier implements Serializable{
    private Topology topology;
    private Instances dataset;
    private enum DeltaRuleType{Batch,Incremental};
    private DeltaRuleType rule = DeltaRuleType.Batch;
    private NominalToBinary nominalToBinaryFilter = new NominalToBinary();
    private Normalize normalizeFilter = new Normalize();

    public DeltaTrainingRule(){
        topology = new Topology();
    }

    public DeltaTrainingRule(Topology t){
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

    public DeltaRuleType getDetlaRuleType(){
        return rule;
    }
    
    public void setDeltaRuleType(DeltaRuleType ruletype){
        this.rule = ruletype;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        dataset = new Instances(data);
        for(int i=0; i<dataset.numAttributes(); i++)
        {
            dataset.deleteWithMissing(i);
        }
        nominalToBinaryFilter.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, nominalToBinaryFilter);
        normalizeFilter.setInputFormat(dataset);
        dataset =Filter.useFilter(dataset,normalizeFilter);
        topology.addInputLayer(dataset.numAttributes() - 1);
        topology.addOutputLayer(1);

        topology.connectAllNodes();

        double epochError = 0;
        int tries = 0;
        Node outputNode = topology.getOutputNode(0);
        outputNode.setBiasWeight(0.1);
        double treshold = topology.isUseErrorThreshold()?topology.getEpochErrorThreshold():0.0;
        do {
            epochError = 0;
            if(rule == DeltaRuleType.Incremental){
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
                    double target = instance.classValue();
                    topology.initInputNodes(instance);

                    topology.resetNodesInput();

                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                    }
                    outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                    double output = outputNode.getInput();
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
                }
                for(int i=0;i<dataset.numInstances();i++){
                    Instance instance = dataset.instance(i);
                    double target = instance.classValue();
                    topology.initInputNodes(instance);
                    topology.resetNodesInput();
                    topology.initInputNodes(instance);
                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                    }
                    outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                    double output = outputNode.getInput();
                    double squaredError = (output-target)*(output-target);
                    epochError += squaredError;
                }
            }
            else if(rule == DeltaRuleType.Batch){
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
                    double target = instance.classValue();
                    topology.initInputNodes(instance);

                    topology.resetNodesInput();
                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                    }
                    outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                    double output = outputNode.getInput();
                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.setPreviousDeltaBatch(weight.getPreviousDeltaBatch() + topology.getLearningRate() * (target - output) * weight.getNode1().getOutput());
                    }

                    outputNode.setDeltabatch(outputNode.getDeltabatch() + (topology.getLearningRate() * (target - output) * outputNode.getBiasValue()));
                }

                for (int j = 0; j < topology.getWeights().size(); j++) {
                    Weight weight = topology.getWeights().get(j);
                    weight.setWeight(weight.getWeight()+weight.getPreviousDeltaBatch() + topology.getMomentumRate()* weight.getPreviousDeltaWeight());
                    weight.setPreviousDeltaWeight(weight.getPreviousDeltaBatch() + topology.getMomentumRate()* weight.getPreviousDeltaWeight());
                    weight.setPreviousDeltaBatch(0.0);
                }

                outputNode.setBiasWeight(outputNode.getBiasWeight() + outputNode.getDeltabatch() + topology.getMomentumRate() * outputNode.getPreviousDeltaWeight());
                outputNode.setPreviousDeltaWeight(outputNode.getDeltabatch() + topology.getMomentumRate()* outputNode.getPreviousDeltaWeight());
                outputNode.setDeltabatch(0.0);


                //Squared n Eppoch Error
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
                    topology.initInputNodes(instance);
                    double target = instance.classValue();
                    double output = 0.0;
                    topology.resetNodesInput();
                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                    }
                    outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                    output = outputNode.getInput();
                    double squaredError = (output-target)*(output-target);
                    epochError += squaredError;
                    topology.resetNodesInput();
                }
            }
            else{
                    throw (new Exception("Ruletype can only be batch or incremental!"));
            }
            tries++;
            epochError *= 0.5;
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
        
        long output = Math.round(outputNode.getInput());
        if(output > dataset.numClasses() - 1){
            output = dataset.numClasses() - 1;
        }
        return (double)output;
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
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    public static void main(String [] args) throws Exception {
        Instances dataset = Util.readARFF("iris.arff");

        Topology topology = new Topology();
        topology.setLearningRate(0.1);
        topology.setInitialWeight(0.0);
        topology.setMomentumRate(0.0);
        topology.setEpochErrorThreshold(0);
        topology.setNumIterations(5);

        DeltaTrainingRule dtr = new DeltaTrainingRule(topology);
        dtr.setDeltaRuleType(DeltaRuleType.Batch);
        dtr.buildClassifier(dataset);

        Evaluation eval = Util.evaluateModel(dtr, dataset);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
        System.out.println(dtr);
    }
}
