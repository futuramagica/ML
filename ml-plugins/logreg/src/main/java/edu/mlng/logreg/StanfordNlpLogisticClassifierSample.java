package edu.mlng.logreg;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LogisticClassifier;
import edu.stanford.nlp.classify.LogisticClassifierFactory;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Example of Stanford NLP LogisticClassifier usage
 */
public class StanfordNlpLogisticClassifierSample {

    public static void main(String[] args) throws Exception {
        test("mlclass-ex2\\ex2data1.txt");
        test("mlclass-ex2\\ex2data2.txt");
    }

    private static void test(String dataFile) throws IOException {
        String file = System.getProperty("mlclass-resources", "") + dataFile;

        Dataset<String, String> ds = new Dataset<>();
        List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());
        List<Datum<String, String>> data = new ArrayList<>(lines.size());
        for (String line : lines) {
            String[] bits = line.split(",");
            Collection<String> features = Arrays.asList(bits).subList(0, bits.length - 1);
            String label = bits[bits.length - 1];
            data.add(new BasicDatum<>(features, label));
        }

        ds.addAll(data);
        ds.summaryStatistics();

        LogisticClassifierFactory<String, String> factory = new LogisticClassifierFactory<>();
        LogisticClassifier<String, String> lc = factory.trainClassifier(ds);

        int mismatches = 0;
        // count mismatches (print first 5)
        for (Datum<String, String> d: data) {
            Collection<String> features = d.asFeatures();
            String guessed = lc.classOf(features);
//            System.out.println(line + "\t => " + g);
            if (!guessed.equals(d.label())) {
                mismatches++;
                if (mismatches < 5)
                    System.out.println("Expected " + d.label() + " but guessed " + guessed +
                            " - score: " + lc.scoreOf(features));
            }
        }

        System.out.println(mismatches + " mismatches found of " + ds.size() +
                " (~" + (mismatches * 100 / ds.size()) + "%)");
    }
}
