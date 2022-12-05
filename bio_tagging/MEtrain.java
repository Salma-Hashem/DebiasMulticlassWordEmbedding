// Wrapper for maximum-entropy training

// NYU - Natural Language Processing - Prof. Grishman

// invoke by:  java  MEtrain  dataFile  modelFile

import java.io.*;
import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;

public class MEtrain {

    public static void main (String[] args) {
	if (args.length != 2) {
	    System.err.println ("MEtrain requires 2 arguments:  dataFile modelFile");
	    System.exit(1);
	}
	String dataFileName = args[0];
	String modelFileName = args[1];
	try {
	    // read events with tab-separated features
	    FileReader datafr = new FileReader(new File(dataFileName));
	    EventStream es = new BasicEventStream(new PlainTextByLineDataStream(datafr), "\t");
	    // train model using 100 iterations, ignoring events occurring fewer than 4 times
        // spenser comment: experimenting with 5 times and 200 iterations
        // so far: best performance with 110&4
        // score tracking: 500&7 f-score: 8.97
	    GISModel model = GIS.trainModel(es, 600, 7);
	    // save model
	    File outputFile = new File(modelFileName);
	    GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
	    writer.persist();
	} catch (Exception e) {
	    System.out.print("Unable to create model due to exception: ");
	    System.out.println(e);
	}
    }
}
