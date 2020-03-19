package cnn;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/*
 * kod zde použitý je původně z knihy Java Deep Learning Cookbook 
 * původní kod na adrese:
 * https://github.com/PacktPublishing/Java-Deep-Learning-Cookbook/blob/master/05_Implementing_NLP/sourceCode/cookbookapp/src/main/java/com/javadeeplearningcookbook/examples/CnnWord2VecSentenceClassificationExample.java
 */

public class CNNwithWord2Vec {
	
    /*
     * String dataFalse, String dataTrue, String dataTestFalse, String dataTestTrue, String wordVectorsPath, int vectorSize, int numOfEpochs
     */
	public void basicsettings(String dataFalse, String dataTrue, String dataTestFalse, String dataTestTrue, String wordVectorsPath, int vectorSize, int numOfEpochs) {
    	
		//Globální nastavení sítě
        int batchSize = 32;
        int truncateReviewsToLength = 256;  

        int cnnLayerFeatureMaps = 100;      
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); 


        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)      
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(2)    
                        .build(), "globalPool")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Loading word vectors and creating DataSetIterators");

        // načtení word2Vec modelu do paměti
        // u google word2Vec vyžaduje asi 6gb RAM paměti
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
        
        //volání metody pro tvorbu iterátorů
        DataSetIterator trainIter = getDataSetIterator(dataFalse, dataTrue, wordVectors, batchSize, truncateReviewsToLength, rng);
        
        DataSetIterator testIter = getDataSetIterator(dataTestFalse, dataTestTrue, wordVectors, batchSize, truncateReviewsToLength, rng);

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < numOfEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //po každé epoše vypíšeme úspěšnost
            Evaluation evaluation = net.evaluate(testIter);

            System.out.println(evaluation.stats());
        }
    }
	
	//metoda pro vytvoření iterátorů 
    private static DataSetIterator getDataSetIterator(String fal, String tru, WordVectors wordVectors, int minibatchSize,
            int maxSentenceLength, Random rng ){
    	
    	//cesty k datům
    	File trueNews = new File(fal);
        File falseNews = new File(tru);
        
        
        Map<String,List<File>> reviewFilesMap = new HashMap<>();
        reviewFilesMap.put("Positive", Arrays.asList(trueNews.listFiles()));
        reviewFilesMap.put("Negative", Arrays.asList(falseNews.listFiles()));
    	
    	LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap, rng);
    	
    	//pro CNN již existuje předvytvořený iterátor
        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .sentenceProvider(sentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    	

    }
}
