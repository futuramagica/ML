package edu.mlng.logreg;

import java.io.InputStream;

public interface SimpleLogisticRegression {

    static interface Classifier {

        int classifyInstance(double[] instance);

        double[] distributionForInstance(double[] instance);
    }

    Classifier buildClassifier(InputStream source);

    Classifier buildClassifier(double[][] data, int[] classes);

    class ClassifierException extends RuntimeException {

        public ClassifierException(Throwable cause) {
            super(cause);
        }

        public ClassifierException(String message) {
            super(message);
        }
    }
}
