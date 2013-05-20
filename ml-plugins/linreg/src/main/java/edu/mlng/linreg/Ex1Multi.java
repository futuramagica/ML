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
public class Ex1Multi {

    public static void main(String[] args) throws IOException {

        // ================ Part 1: Feature Normalization ================

        Matrix data = new Matrix(fromFile("ex1data2.txt"));
        int m = data.getRowDimension();
        Matrix x = data.getMatrix(0, m - 1, 0, 1);
        Matrix y = data.getMatrix(0, m - 1, 2, 2);

        log("First 10 examples from the dataset: ");
        log(" x =    \n" + x.getMatrix(0, 9, 0, 1).transpose());
        log(" y =    \n" + y.getMatrix(0, 9, 0, 0).transpose());

        featureNormalize(x);

        // Add intercept term to X
        Matrix X = new Matrix(m, 3);
        X.setMatrix(0, m - 1, 0, 0, new Matrix(m, 1, 1.0));
        X.setMatrix(0, m - 1, 1, 2, x);


        // ================ Part 2: Gradient Descent ================

        int iterations = 1500;
        double alpha = 0.01;

        Matrix theta = new Matrix(3, 1);
        theta = gradientDescentMulti(X, y, theta, alpha, iterations);

        // Display gradient descent's result
        log("Theta computed from gradient descent: " + theta.transpose());


        // ================ Part 3: Normal Equations ================

        log("Solving with normal equations...");

        theta = normalEqn(X, y);

        log("Theta computed from the normal equations: " + theta.transpose());
    }

    private static Matrix normalEqn(Matrix x, Matrix y) {
        Matrix xt = x.transpose();
        return xt.times(x).inverse().times(xt).times(y);
    }

    private static void featureNormalize(Matrix x) {
        //To change body of created methods use File | Settings | File Templates.
    }

    private static Double computeCost(Matrix x, Matrix y, Matrix theta) {
        Matrix d = x.times(theta).minus(y);
        Matrix sum = d.transpose().times(d);
        int m = y.getRowDimension();
        return 1.0 / (2 * m) * sum.get(0, 0);
    }

    private static Matrix gradientDescentMulti(Matrix x, Matrix y, Matrix theta, double alpha, int iterations) {
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
