package edu.mlng.logreg;

import weka.classifiers.functions.SimpleLogistic;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimpleLogisticRegressionWeka implements SimpleLogisticRegression {

    public static void main(String[] args) throws Exception {
        test("mlclass-ex2\\ex2data1.txt");
        test("mlclass-ex2\\ex2data2.txt");
    }

    static void test(String dataFile) throws Exception {

        String path = System.getProperty("mlclass-resources", "") + dataFile;
        Classifier classifier;
        try (InputStream inputStream = new FileInputStream(path)) {
            SimpleLogisticRegressionWeka logisticRegression = new SimpleLogisticRegressionWeka();
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

    public final Classifier buildClassifier(InputStream source) {
        String separator = ",";
        BufferedReader reader = new BufferedReader(new InputStreamReader(source));
        String line;
        try {
            int numAttributes = -1;
            List<double[]> data = new ArrayList<>();
            List<Integer> classes = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(separator);
                if (numAttributes == -1)
                    numAttributes = values.length - 1;
                else if (numAttributes != values.length - 1)
                    throw new ClassifierException("Error building instances, wrong line " + data.size());
                double[] attValues = new double[numAttributes];
                for (int i = 0; i < numAttributes; i++)
                    attValues[i] = Double.parseDouble(values[i]);
                data.add(attValues);
                classes.add(Integer.parseInt(values[numAttributes]));
            }
            return buildClassifier(data.toArray(new double[data.size()][]), toPrimitiveArray(classes));
        } catch (IOException e) {
            throw new ClassifierException(e);
        }
    }

    private static int[] toPrimitiveArray(List<Integer> integers) {
        int[] result = new int[integers.size()];
        for (int i = 0; i < result.length; i++)
            result[i] = integers.get(i);
        return result;
    }

    @Override
    public Classifier buildClassifier(double[][] data, int[] classes) {
        if (data.length == 0 || data.length != classes.length || data[0].length == 0)
            throw new IllegalArgumentException();
        final Instances instances = dataToInstances(data, classes);
        return new ClassifierImpl(instances);
    }

    private Instances dataToInstances(double[][] data, int[] classes) {
        int numInstances = data.length;
        int numAttributes = data[0].length + 1;
        FastVector attInfo = new FastVector(numAttributes);
        for (int i = 0; i < numAttributes - 1; i++)
            attInfo.addElement(new Attribute("" + i));

        FastVector classValues = new FastVector();
        classValues.addElement("0");
        classValues.addElement("1");
        Attribute classAttribute = new Attribute("class", classValues);
        attInfo.addElement(classAttribute);
        Instances instances = new Instances(null, attInfo, numInstances);
        instances.setClassIndex(numAttributes - 1);
        for (int i = 0; i < numInstances; i++) {
            Instance instance = dataToInstance(data[i]);
            instance.setDataset(instances);
            instance.setClassValue(classes[i]);
            instances.add(instance);
        }
        return instances;
    }

    static Instance dataToInstance(double[] data) {
        return new Instance(1, Arrays.copyOf(data, data.length + 1));
    }

    private static class ClassifierImpl implements Classifier {

        private final SimpleLogistic classifier;

        public ClassifierImpl(Instances instances) {
            classifier = new SimpleLogistic();
            try {
                classifier.buildClassifier(instances);
            } catch (Exception e) {
                throw new ClassifierException(e);
            }
        }

        @Override
        public int classifyInstance(double[] instance) {
            return maxIndex(distributionForInstance(instance));
        }

        @Override
        public double[] distributionForInstance(double[] instance) {
            try {
                return classifier.distributionForInstance(dataToInstance(instance));
            } catch (Exception e) {
                throw new ClassifierException(e);
            }
        }

        private int maxIndex(double[] dist) {
            double max = 0;
            int maxIndex = 0;
            for (int i = 0; i < dist.length; i++)
                if (dist[i] > max) {
                    maxIndex = i;
                    max = dist[i];
                }
            return maxIndex;
        }
    }
}
