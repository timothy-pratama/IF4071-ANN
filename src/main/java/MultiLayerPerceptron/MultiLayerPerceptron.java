package MultiLayerPerceptron;

import Util.Util;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by timothy.pratama on 07-Nov-15.
 */
public class MultiLayerPerceptron extends Classifier {
    private Instances dataset;
    private HashMap<Integer, HashMap<Integer, Double>> weights;
    private ArrayList<Node> nodes;
    private int inputNeuronSize;
    private int outputNeuronSize;
    private int hiddenLayerSize[];
    private int nodesSize;
    private double initialWeight;
    private boolean randomWeight;

    public MultiLayerPerceptron() {
        weights = new HashMap<>();
        inputNeuronSize = 0;
        outputNeuronSize = 0;
        hiddenLayerSize = new int[0];
        nodesSize = 0;
        nodes = new ArrayList<>();
        randomWeight = true;
        initialWeight = 0;
    }

    public void setInputNeuronSize(int size) {
        inputNeuronSize = size;
    }

    public void setOutputNeuronSize(int size) {
        outputNeuronSize = size;
    }

    public void setHiddenLayerSize(int size) {
        hiddenLayerSize = new int[size];
    }

    public void setHiddenNeuronSize(int hiddenLayer, int size) {
        hiddenLayerSize[hiddenLayer] = size;
    }

    public void setInitialWeight(double weight)
    {
        initialWeight = weight;
        randomWeight = false;
    }

    public void printNodeRelation()
    {
        System.out.println("===== Node Relation =====");
        for(HashMap.Entry<Integer, HashMap<Integer, Double>> from : weights.entrySet())
        {
            System.out.printf("Node %d\n", from.getKey());
            for(HashMap.Entry<Integer, Double> to : from.getValue().entrySet())
            {
                System.out.printf("|=>%d = %f\n", to.getKey(), to.getValue());
            }
        }
        System.out.println();
    }

    public void printNodeBias()
    {
        System.out.println("===== Node Bias =====");
        for(int i=inputNeuronSize; i<nodesSize; i++)
        {
            System.out.printf("Bias[%d]: %f\n",i,nodes.get(i).getBias());
        }
        System.out.println();
    }

    private double getRandom(double min, double max)
    {
        double value = 0.0;
        Random r = new Random();
        value = min + (max-min) * r.nextDouble();
        return value;
    }

    private void init()
    {
        nodesSize = inputNeuronSize + outputNeuronSize;
        HashMap<Integer, Double> temp;
        HashMap<Integer, Double> temp2;
        double value;
        int start_idx;

        for(int i=0; i<hiddenLayerSize.length; i++)
        {
            nodesSize += hiddenLayerSize[i];
        }

        for(int i=0; i<nodesSize; i++)
        {
            Node tempNode = new Node();
            if(randomWeight)
            {
                tempNode.setBias(getRandom(-0.1, 0.1));
            }
            else
            {
                tempNode.setBias(initialWeight);
            }
            nodes.add(tempNode);
        }

        if(hiddenLayerSize.length > 0)
        {
            for (int i = 0; i < inputNeuronSize; i++)
            {
                for (int j = inputNeuronSize; j < inputNeuronSize + hiddenLayerSize[0]; j++)
                {
                    if (randomWeight) {
                        value = getRandom(-0.1, 0.1);
                    } else {
                        value = initialWeight;
                    }
                    temp2 = weights.get(i);
                    if(temp2 == null)
                    {
                        temp = new HashMap<>();
                        temp.put(j, value);
                        weights.put(i, temp);
                    }
                    else
                    {
                        temp2.put(j, value);
                    }
                }
            }

            start_idx = inputNeuronSize;
            for(int j=0; j<hiddenLayerSize.length - 1; j++)
            {
                for(int k=0; k<hiddenLayerSize[j]; k++)
                {
                    for(int l=0; l<hiddenLayerSize[j+1]; l++)
                    {
                        if(randomWeight)
                        {
                            value = getRandom(-0.1, 0.1);
                        }
                        else
                        {
                            value = initialWeight;
                        }
                        temp2 = weights.get(start_idx+k);
                        if(temp2 == null)
                        {
                            temp = new HashMap<>();
                            temp.put(start_idx+hiddenLayerSize[j]+l, value);
                            weights.put(start_idx + k, temp);
                        }
                        else
                        {
                            temp2.put(start_idx+hiddenLayerSize[j]+1, value);
                        }

                    }
                }
                start_idx = start_idx + hiddenLayerSize[j];
            }

            for(int j=0; j<hiddenLayerSize[hiddenLayerSize.length-1]; j++)
            {
                for(int k=0; k<outputNeuronSize; k++)
                {
                    if(randomWeight)
                    {
                        value = getRandom(-0.1, 0.1);
                    }
                    else
                    {
                        value = initialWeight;
                    }
                    temp2 = weights.get(start_idx+j);
                    if(temp2 == null)
                    {
                        temp = new HashMap<>();
                        temp.put(start_idx + hiddenLayerSize[hiddenLayerSize.length - 1], value);
                        weights.put(start_idx+j, temp);
                    }
                    else
                    {
                        temp2.put(start_idx + hiddenLayerSize[hiddenLayerSize.length - 1], value);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < inputNeuronSize; i++)
            {
                for (int j = inputNeuronSize; j < inputNeuronSize + outputNeuronSize; j++)
                {
                    if (randomWeight) {
                        value = getRandom(-0.1, 0.1);
                    } else {
                        value = initialWeight;
                    }
                    temp2 = weights.get(i);
                    if(temp2 == null)
                    {
                        temp = new HashMap<>();
                        temp.put(j, value);
                        weights.put(i, temp);
                    }
                    else
                    {
                        temp2.put(j, value);
                    }
                }
            }
        }
    }

    @Override
    public void buildClassifier(Instances data){
        dataset = new Instances(data);
        init();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        // Attributes capabilities
        capabilities.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.MISSING_VALUES);

        // Class Capabilities
        capabilities.enable(Capabilities.Capability.NOMINAL_CLASS);
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);
        capabilities.enable(Capabilities.Capability.DATE_CLASS);
        capabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return capabilities;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    public static void main (String[] args)
    {
        Instances dataset = Util.readARFF("weather.nominal.arff");
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron();
        multiLayerPerceptron.setInputNeuronSize(3);
        multiLayerPerceptron.setOutputNeuronSize(1);
        multiLayerPerceptron.setHiddenLayerSize(2);
        multiLayerPerceptron.setHiddenNeuronSize(0, 2);
        multiLayerPerceptron.setHiddenNeuronSize(1,2);
        multiLayerPerceptron.setInitialWeight(0.0);
        multiLayerPerceptron.buildClassifier(dataset);

        multiLayerPerceptron.printNodeRelation();
        multiLayerPerceptron.printNodeBias();
    }
}
