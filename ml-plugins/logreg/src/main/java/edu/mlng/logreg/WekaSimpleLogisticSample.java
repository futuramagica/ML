package edu.mlng.logreg;

import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.Arrays;
import java.util.Enumeration;

/**
 * Example of Weka SimpleLogistic classifier usage
 */
public class WekaSimpleLogisticSample {

    public static void main(String[] args) throws Exception {
        test("mlclass-ex2\\ex2data1.txt");
        test("mlclass-ex2\\ex2data2.txt");
    }

    static void test(String dataFile) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(System.getProperty("mlclass-resources", ""), dataFile));
        loader.setNominalAttributes("last");
        Instances instances = loader.getDataSet();

        instances.setClassIndex(instances.numAttributes() - 1);
        SimpleLogistic classifier = new SimpleLogistic();
        classifier.buildClassifier(instances);

        int mismatches = 0;
        // count mismatches (print first 5)
        Enumeration enumeration = instances.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            double guessed = classifier.classifyInstance(instance);
            if (guessed != instance.classValue()) {
                mismatches++;
                if (mismatches < 5)
                    System.out.println("Expected " + instance.classValue() + " but guessed " + guessed +
                            " - distribution: " + Arrays.toString(classifier.distributionForInstance(instance)));
            }
        }
        System.out.println(mismatches + " mismatches found of " + instances.numInstances() +
                " (~" + (mismatches * 100 / instances.numInstances()) + "%)");
    }
}
