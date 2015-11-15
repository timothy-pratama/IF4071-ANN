package Model;

/**
 * Created by timothy.pratama on 15-Nov-15.
 */
public class Node {
    public static double siegmoid(double x)
    {
        return 1 / (1 + (Math.exp(-x)));
    }

    public static int sign(double x)
    {
        if(x > 0)
        {
            return 1;
        }
        else
        {
            return -1;
        }
    }

    public static void testOutputFunction()
    {
        System.out.printf("Siegmoid(5): %f\n", Node.siegmoid(5));
        System.out.printf("Sign(0): %d\n", Node.sign(0));
    }

    public static void main (String [] args)
    {
        
    }
}
