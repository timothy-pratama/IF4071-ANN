package MultiLayerPerceptron;

import Model.Topology;
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
        System.out.println(dataset);
        topology.addInputLayer(dataset.numAttributes()-1);
        topology.addOutputLayer(dataset.numClasses());
    }

    @Override
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    public static void main(String [] args) throws Exception {
        Instances dataset = Util.readARFF("weather.nominal.arff");
        Topology topology = new Topology();
        topology.addHiddenLayer(1);
        topology.addHiddenLayer(2);
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(topology);
        mlp.buildClassifier(dataset);
    }
}
