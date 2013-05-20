package edu.mlng.linreg;

import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class LinearRegression {

    public double[] gradientDescent(InputStream source, double alpha, int iterations) {
        return gradientDescent(fromInputStream(source), alpha, iterations);
    }

    public double[] gradientDescent(double[][] data, double alpha, int iterations) {
        int m = data.length;
        Matrix dataMatrix = new Matrix(data);
        Matrix x = new Matrix(m, 2);
        x.setMatrix(0, m - 1, 0, 0, new Matrix(m, 1, 1.0));
        x.setMatrix(0, m - 1, 1, 1, dataMatrix.getMatrix(0, m - 1, 0, 0));
        Matrix y = dataMatrix.getMatrix(0, m - 1, 1, 1);
        Matrix theta = new Matrix(2, 1);
        for (int i = 0; i < iterations; i++) {
            Matrix d = x.times(theta).minus(y);
            Matrix sum = d.transpose().times(x);
            theta.minusEquals(sum.times(alpha / m).transpose());
        }
        return theta.transpose().getArray()[0];
    }

    public static void main(String[] args) throws IOException {
        String resources = "ML\\ml-plugins\\linreg\\src\\main\\resources";
        Path path = Paths.get(resources, "ex1data1.txt");
        InputStream inputStream = Files.newInputStream(path);

        LinearRegression linearRegression = new LinearRegression();
        int iterations = 1500;
        double alpha = 0.01;
        double[] theta = linearRegression.gradientDescent(
                inputStream, alpha, iterations);

        System.out.println("Theta found by gradient descent: " + theta[0] + " " + theta[1]);

        double predict1 = innerProduct(theta, new double[]{1, 3.5});
        System.out.println("For population = 35,000, we predict a profit of " + predict1 * 10000);

        double predict2 = innerProduct(theta, new double[]{1, 7});
        System.out.println("For population = 70,000, we predict a profit of " + predict2 * 10000);
    }

    private static double innerProduct(double[] v1, double[] v2) {
        return new DoubleVector(v2).innerProduct(new DoubleVector(v1));
    }

    private static double[][] fromInputStream(InputStream inputStream) {
        List<double[]> data = new ArrayList<>();
        try {
            BufferedReader reader =
                    new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null)
                data.add(toDoubles(line.split(",")));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data.toArray(new double[data.size()][]);
    }

    private static double[] toDoubles(String[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++)
            result[i] = Double.parseDouble(values[i]);
        return result;
    }
}
