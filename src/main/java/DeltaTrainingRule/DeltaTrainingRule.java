package DeltaTrainingRule;

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
            if(rule == DeltaRuleType.Incremental){
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
                    double target = instance.classValue();
                    topology.initInputNodes(instance);

                    //topology.sortWeight(false, true);
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
                        topology.resetNodesInput();
                        for (int j = 0; j < topology.getWeights().size(); j++) {
                            Weight weight = topology.getWeights().get(j);
                            weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                        }
                        outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                        output = outputNode.getInput();
                        double squaredError = (output-target)*(output-target);
                        epochError += squaredError;
                    }
                }
            else if(rule == DeltaRuleType.Batch){
                double delta = 0.0f;
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
                    double target = instance.classValue();
                    topology.initInputNodes(instance);

                    //topology.sortWeight(false, true);
                    topology.resetNodesInput();

                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
                    }
                    outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
                    double output = outputNode.getInput();
                    for (int j = 0; j < topology.getWeights().size(); j++) {
                        Weight weight = topology.getWeights().get(j);
                        weight.setDeltabatch(weight.getDeltabatch() + (topology.getLearningRate() * (target - output) * weight.getNode1().getOutput()));
                    }
                    outputNode.setDeltabatch(outputNode.getDeltabatch() + (topology.getLearningRate() * (target - output) * outputNode.getBiasValue()));
                        
                }               
                for (int j = 0; j < topology.getWeights().size(); j++) {
                    Weight weight = topology.getWeights().get(j);
                    weight.setWeight(weight.getWeight()+weight.getDeltabatch() + topology.getMomentumRate()* weight.getPreviousDeltaWeight());
                    weight.setPreviousDeltaWeight(weight.getDeltabatch() + topology.getMomentumRate()* weight.getPreviousDeltaWeight());
                    weight.setDeltabatch(0.0);
                }
                outputNode.setBiasWeight(outputNode.getBiasWeight() + outputNode.getDeltabatch() + topology.getMomentumRate()* outputNode.getPreviousDeltaWeight());
                outputNode.setPreviousDeltaWeight(outputNode.getDeltabatch() + topology.getMomentumRate()* outputNode.getPreviousDeltaWeight());
                outputNode.setDeltabatch(0.0);
                
                for (int i = 0; i < dataset.numInstances(); i++) {
                    Instance instance = dataset.instance(i);
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
                }
            }
            else{
                    throw (new Exception("Ruletype can only be batch or incremental!"));
            }
            tries++;
            epochError *= 0.5f;
            System.out.println("Iteration "+tries+" : "+epochError);
        }
        while((epochError > treshold) && (!topology.isUseIteration() || (tries < topology.getNumIterations())));

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        topology.initInputNodes(instance);
        Node outputNode = topology.getOutputNode(0);
        topology.resetNodesInput();

        for (int j = 0; j < topology.getWeights().size(); j++) {
            Weight weight = topology.getWeights().get(j);
            weight.getNode2().setInput(weight.getNode2().getInput() + (weight.getNode1().getOutput() * weight.getWeight()));
        }
        outputNode.setInput(outputNode.getInput() + (outputNode.getBiasValue() * outputNode.getBiasWeight()));
        
        long output = Math.round(outputNode.getInput());
        System.out.println("Double : "+ output);
        return (double)output;
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
        topology.setUseErrorThreshold(true);
        topology.setUseIteration(true);
        topology.setNumIterations(10000);
        DeltaTrainingRule dtr = new DeltaTrainingRule(topology);
        dtr.setDeltaRuleType(DeltaRuleType.Batch);
        dtr.buildClassifier(dataset);
        Util.classify("iris.numeric.classify.arff",dtr);

    }
}
