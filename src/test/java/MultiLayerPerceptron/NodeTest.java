package MultiLayerPerceptron;

import org.junit.Test;

import static org.junit.Assert.*;

public class NodeTest {

    @Test
    public void testComputeOutput() throws Exception {
        Node node = new Node();
        node.computeOutput(-0.05625);
        assertEquals("Siegmoid",0.48541, node.getOutput(), 0.001);
    }
}