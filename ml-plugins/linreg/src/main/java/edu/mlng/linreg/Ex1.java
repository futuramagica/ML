package edu.mlng.linreg;

import weka.core.matrix.Matrix;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 ** Machine Learning Online Class - Exercise 1: Linear Regression
 *  Instructions
 *  ------------
 * 
 *  This file contains code that helps you get started on the
 *  linear exercise. You will need to complete the following functions 
 *  in this exercise:
 *
 *     warmUpExercise.m
 *     plotData.m
 *     gradientDescent.m
 *     computeCost.m
 *     gradientDescentMulti.m
 *     computeCostMulti.m
 *     featureNormalize.m
 *     normalEqn.m
 *
 *  For this exercise, you will not need to change any code in this file,
 *  or any other files other than those mentioned above.
 *
 * x refers to the population size in 10,000s
 * y refers to the profit in $10,000s
 *
 */
public class Ex1 {

    public static void main(String[] args) throws IOException {

        // ==================== Part 1: Basic Function ====================

        // ======================= Part 2: Plotting =======================
        // TODO implement with JFreeChart

        Matrix data = new Matrix(fromFile("ex1data1.txt"));
        int m = data.getRowDimension();
        Matrix x = data.getMatrix(0, m - 1, 0, 0);
        Matrix y = data.getMatrix(0, m - 1, 1, 1);

        // =================== Part 3: Gradient descent ===================
        Matrix X = new Matrix(m, 2);
        X.setMatrix(0, m - 1, 0, 0, new Matrix(m, 1, 1.0));
        X.setMatrix(0, m - 1, 1, 1, x);
        Matrix theta = new Matrix(2, 1);

        int iterations = 1500;
        double alpha = 0.01;

        log("cost= " + computeCost(X, y, theta));

        theta = gradientDescent(X, y, theta, alpha, iterations);

        log("Theta found by gradient descent: " +
                theta.get(0, 0) + " " + theta.get(1, 0));

        double[][] query1 = {{1, 3.5}};
        double predict1 = new Matrix(query1).times(theta).get(0, 0);
        log("For population = 35,000, we predict a profit of " +  predict1 * 10000);

        double[][] query2 = {{1, 7}};
        double predict2 = new Matrix(query2).times(theta).get(0, 0);
        log("For population = 70,000, we predict a profit of " +  predict2 * 10000);

        // ============= Part 4: Visualizing J(theta_0, theta_1) =============
        // TODO

    }

    private static Double computeCost(Matrix x, Matrix y, Matrix theta) {
        Matrix d = x.times(theta).minus(y);
        Matrix sum = d.transpose().times(d);
        int m = y.getRowDimension();
        return 1.0 / (2 * m) * sum.get(0, 0);
    }

    private static Matrix gradientDescent(Matrix x, Matrix y, Matrix theta, double alpha, int iterations) {
        int m = y.getRowDimension();
        for (int i = 0; i < iterations; i++) {
            Matrix d = x.times(theta).minus(y);
            Matrix sum = d.transpose().times(x);
            theta.minusEquals(sum.times(alpha / m).transpose());
        }
        return theta;
    }


    private static double[][] fromFile(String file) throws IOException {
        String resources = "ML\\ml-plugins\\linreg\\src\\main\\resources";
        Path path = Paths.get(resources, file);
        List<String> lines = Files.readAllLines(path, Charset.defaultCharset());
        int numInstances = lines.size();
        double[][] data = new double[numInstances][];
        for (int i = 0; i < numInstances; i++)
            data[i] = toDoubles(lines.get(i).split(","));
        return data;
    }

    private static double[] toDoubles(String[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++)
            result[i] = Double.parseDouble(values[i]);
        return result;
    }

    private static void log(String message) {
        System.out.println(message);
    }
}
