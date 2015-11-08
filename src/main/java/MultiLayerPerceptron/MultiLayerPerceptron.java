package MultiLayerPerceptron;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by timothy.pratama on 07-Nov-15.
 */
public class MultiLayerPerceptron extends Classifier {
    @Override
    public void buildClassifier(Instances data) throws Exception {

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
}
