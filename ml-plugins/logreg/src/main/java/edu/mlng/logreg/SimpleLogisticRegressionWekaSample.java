package edu.mlng.logreg;

import java.io.FileInputStream;
import java.io.InputStream;
import java.lang.String;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Example of Weka SimpleLogisticRegression implementation usage
 */
public class SimpleLogisticRegressionWekaSample {

    public static void main(String[] args) throws Exception {
        test("mlclass-ex2\\ex2data1.txt");
        test("mlclass-ex2\\ex2data2.txt");
    }

    static void test(String dataFile) throws Exception {

        String path = System.getProperty("mlclass-resources", "") + dataFile;
        SimpleLogisticRegression.Classifier classifier;
        try (InputStream inputStream = new FileInputStream(path)) {
            SimpleLogisticRegression logisticRegression = new SimpleLogisticRegressionWeka();
            classifier = logisticRegression.buildClassifier(inputStream);
        }

        int mismatches = 0;
        // count mismatches (print first 5)
        List<String> lines = Files.readAllLines(Paths.get(path), Charset.defaultCharset());
        int numInstances = lines.size();
        for (String line : lines) {
            String[] values = line.split(",");
            double[] instance = new double[values.length - 1];
            for (int i = 0; i < values.length - 1; i++)
                instance[i] = Double.parseDouble(values[i]);
            int expectedClass = Integer.parseInt(values[values.length - 1]);
            int guessed = classifier.classifyInstance(instance);
            if (guessed != expectedClass) {
                mismatches++;
                if (mismatches < 5)
                    System.out.println("Expected " + expectedClass + " but guessed " + guessed +
                            " - distribution: " + Arrays.toString(classifier.distributionForInstance(instance)));
            }
        }
        System.out.println(mismatches + " mismatches found of " + numInstances +
                " (~" + (mismatches * 100 / numInstances) + "%)");
    }
}
