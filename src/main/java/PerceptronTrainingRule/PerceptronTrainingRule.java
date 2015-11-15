package PerceptronTrainingRule;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by timothy.pratama on 07-Nov-15.
 */
public class PerceptronTrainingRule extends Classifier {
    @Override
    public void buildClassifier(Instances data) throws Exception {

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }
}
