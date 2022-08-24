package javaai.metah.sa;

import javaai.metah.framework.Objective;

public class XorSaObjective implements Objective{
    public final static double RANGE_MAX = 10.0;
    public final static double RANGE_MIN = -10.0;

    /**
     * The input data for XOR
     */
    public static double XOR_INPUTS[][] = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };

    /**
     * The ideal data for XOR.
     */
    public static double XOR_IDEALS[][] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    /**
     * Set ideal inputs and get outputs and fitness
     */
    public static void main(String[] args) {
        double[] best = {-10.04064, 8.86085, 9.95461, -8.87248, 9.97014, 10.08293, -4.71024, -4.99293};
        double[][] outputs = new double[XOR_IDEALS.length][XOR_IDEALS[0].length];

        XorSaObjective XorSa = new XorSaObjective();

        //get the outputs from ideal inputs
        for(int i = 0; i < XOR_INPUTS.length; i++){
            double[] values = XorSa.feedforward(XOR_INPUTS[i], best);
            for(int j = 0; j < values.length; j++) {
                outputs[i][j] = values[j];
            }
        }

        //Get the fitness of ideal inputs
        double fitness = XorSa.getFitness(best);

        //output results
        output(best, XOR_INPUTS, XOR_IDEALS, outputs, fitness);
    }

    /**
     *
     * @param ws  objective function weights
     * @param inputs input values
     * @param ideals ideal outputs
     * @param outputs actual outputs
     * @param fitness RMSE fitness
     */
    public static void output(double[] ws, double[][]inputs, double[][]ideals, double[][] outputs, double fitness){
        System.out.println("  x1   x2   t1   y1");
        for(int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[i].length; j++) {
                System.out.printf("%5.1f", inputs[i][j]);
            }
            for (int k = 0; k < ideals[i].length; k++) {
                System.out.printf("%5.1f", ideals[i][k]);
            }
            for (int p = 0; p < outputs[i].length; p++) {
                System.out.printf("%10.6f", outputs[i][p]);
            }
            System.out.println();
        }
        System.out.print("Best = ");
        for(int i = 0; i < ws.length; i++){
            System.out.printf("%8.3f", ws[i]);
        }
        System.out.println("   fitness = " + fitness);
    }


    /**
     * Prints the results from inputted weights
     * @param ws  objective function weights
     */
    public void print(double[] ws){
        double fitness = getFitness(ws);
        double outputs[][] = outputs(ws);
        System.out.println("  x1   x2   t1   y1");
        for(int i = 0; i < XOR_INPUTS.length; i++) {
            for (int j = 0; j < XOR_INPUTS[i].length; j++) {
                System.out.printf("%5.1f", XOR_INPUTS[i][j]);
            }
            for (int k = 0; k < XOR_IDEALS[i].length; k++) {
                System.out.printf("%5.1f", XOR_IDEALS[i][k]);
            }
            for (int p = 0; p < outputs[i].length; p++) {
                System.out.printf("%10.6f", outputs[i][p]);
            }
            System.out.println();
        }
        System.out.print("Best = ");
        for(int i = 0; i < ws.length; i++){
            System.out.printf("%8.3f", ws[i]);
        }
        System.out.printf("   fitness = %1.8f", fitness);
        System.out.println();
    }

    /**
     * Calculates the outputs from inputted weights
     * @param best weights inputted
     * @return 2d array of outputs
     */
    public double[][] outputs(double[] best){
        double[][] outputs = new double[XOR_IDEALS.length][XOR_IDEALS[0].length];
        for(int i = 0; i < XOR_INPUTS.length; i++){
            double[] values = feedforward(XOR_INPUTS[i], best);
            for(int j = 0; j < values.length; j++) {
                outputs[i][j] = values[j];
            }
        }
        return outputs;
    }


    /**
     * Gets the fitness of weights.
     * @param ws Weights
     * @return Fitness value
     */
    public double getFitness(double[] ws) {
        // Sum of square error
        double sumSqrErr = 0.0;

        for(int k = 0; k < XOR_INPUTS.length; k++) {
            double x1 = XOR_INPUTS[k][0];
            double x2 = XOR_INPUTS[k][1];
            double[] xs = {x1, x2};

            double[] actual = feedforward(xs, ws);
            double ideal = XOR_IDEALS[k][0];

            // Square error
            double sqrError = (actual[0] - ideal) * (actual[0] - ideal);

            // Sum the square error
            sumSqrErr += sqrError;
        }

        double rmse = Math.sqrt(sumSqrErr / XOR_INPUTS.length);

        return rmse;
    }

    /**
     * Runs the inputs through the feedforward equations.
     * @param xs x1 and x2 input
     * @param ws Weights w1-w6 and b1 and b2 -- in this order.
     * @return Actual, that is, Y1
     */
    public double[] feedforward(double[] xs, double[] ws) {
        double w1 = ws[0];
        double w2 = ws[1];
        double w3 = ws[2];
        double w4 = ws[3];
        double w5 = ws[4];
        double w6 = ws[5];
        double b1 = ws[6];
        double b2 = ws[7];

        double zh1 = w1*xs[0] + w3*xs[1] + b1;
        double zh2 = w2*xs[0] + w4*xs[1] + b1;
        double h1 = sigmoid(zh1);
        double h2 = sigmoid(zh2);
        double zy1 = w5*h1 + w6*h2 + b2;
        double y1 = sigmoid(zy1);

        double[] values = {y1};

        return values;
    }

    /**
     * Sigmoid activation function
     * @param z Input
     * @return sigmoid of z
     */
    protected double sigmoid(double z) {
        return 1.0 / (1+ Math.exp(-z));
    }

    public double getRandomWeight() {
        double wt = Math.random()*(RANGE_MAX-RANGE_MIN)+RANGE_MIN;
        return wt;
    }



}
