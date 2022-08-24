package javaai.metah.sa;

import javaai.metah.framework.Heuristic;
import javaai.metah.framework.Objective;

import java.util.Random;

public class XorSa implements Heuristic {
    Random r = new Random();

    XorSaObjective XorSaObjective;

    double[] best_weights;
    double[] current_weights;
    double[] candidate_weights;

    //fitness of different weight arrays
    double candidate;
    double current;
    double best;

    //batch size 0 if full batch
    int batch;
    //batch counter
    int cur_batch;
    //epoch counter
    int epoch;

    //Bound of weights
    public static final double BOUND = 20;
    //learning rate 20-30 works best
    public static final double LEARNING_RATE = 25;
    //rate of cooling from 1000
    public static final double COOLING_FACTOR = 0.9999;

    /**
     * Constructor
     */
    XorSa(){
        XorSaObjective = new XorSaObjective();

        best_weights = new double[8];
        current_weights = new double[8];
        candidate_weights = new double[8];

        candidate = Double.MAX_VALUE;
        current = Double.MAX_VALUE;
        best = Double.MAX_VALUE;

        epoch = 0;
        cur_batch = 0;
    }


    public static void main(String[] args) {
        XorSa sa = new XorSa();

        sa.learn(sa.XorSaObjective);
    }

    /**
     * Main training function
     * @param obj Objective learning algorithm employs
     * @return
     */
    @Override
    public double[] learn(Objective obj) {
        setBatch(0);
        double temperature =  .1;
        double stop = temperature/1000;

        init();

        for (double t = temperature; t > stop; t *= COOLING_FACTOR) {
            epoch++;
            evolve();

            if(candidate < best){
                best_weights = candidate_weights.clone();
                best = candidate;
                //output
                XorSaObjective.print(best_weights);
                System.out.println("Epoch: " + epoch);
                System.out.println("Temperature: " + t);
                System.out.println();
            }

            //Probability function
            double probability = Math.pow(Math.E, -(candidate - current)/t);
            if(probability > Math.random()){
                current_weights = candidate_weights.clone();
                current = candidate;
            }
            if(best < 0.01) break;
        }

        System.out.println("Final Results");
        System.out.println("----------------------------------");
        XorSaObjective.print(best_weights);
        System.out.println("Epoch: " + epoch);

        System.out.println();

        return best_weights;
    }


    /**
     * Initializes first weights used
     */
    private void init() {
        for(int i = 0; i < current_weights.length; i++){
            current_weights[i] = XorSaObjective.getRandomWeight();
        }
        current = XorSaObjective.getFitness(current_weights);
    }

    /**
     * Runs one cycle of the heuristic.
     * <p>It is invoked by <i>learn</i> method.</p>
     * @return Fitness of the current solution.
     */
    @Override
    public double evolve() {
        candidate = current;
        candidate_weights = current_weights.clone();

        double new_weight;
        if(batch == 0){
            for(int i = 0; i < candidate_weights.length; i++) {
                //agitate the current weight
                new_weight = candidate_weights[i] + Math.random()*candidate*LEARNING_RATE*(r.nextBoolean()?1:-1);
                if(new_weight < BOUND && new_weight > -BOUND){
                    candidate_weights[i] = new_weight;
                }
            }
        }
        else{
            for(int i = 0; i < batch; i++) {
                //agitate current weight
                new_weight = candidate_weights[i] + Math.random()*candidate*LEARNING_RATE*(r.nextBoolean()?1:-1);
                if(new_weight < BOUND && new_weight > -BOUND){
                    candidate_weights[(cur_batch)%candidate_weights.length] = new_weight;
                }
                cur_batch++;
            }
        }

        //get candidate fitness and return
        candidate = XorSaObjective.getFitness(candidate_weights);
        return candidate;
    }


    /**
     * Gets the best weights so far
     * @return
     */
    @Override
    public double[] getBest() {
        return best_weights;
    }


    /**
     * Gets the current weights
     * @return
     */
    @Override
    public double[] getCurrent() {
        return current_weights;
    }


    /**
     * Sets the batch size 0-8
     * @param size
     */
    @Override
    public void setBatch(int size) {
        batch = size;
    }
}
