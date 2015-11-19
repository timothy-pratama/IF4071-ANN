package DeltaTrainingRule;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 * Created by timothy.pratama on 07-Nov-15.
 */
public class DeltaTrainingRule extends Classifier implements Serializable{
    @Override
    public void buildClassifier(Instances data) throws Exception {

    }

    @Override
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }
}
