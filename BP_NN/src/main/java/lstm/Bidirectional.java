package lstm;

import java.io.File;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Bidirectional {
	
	/*
     * String dataFalse, String dataTrue, String dataTestFalse, String dataTestTrue, String wordVectorsPath, int vectorSize, int numOfEpochs
     */
	
	public void basicsettings(String dataFalse, String dataTrue, String dataTestFalse, String dataTestTrue, String wordVectorsPath, int vectorSize, int numOfEpochs) throws Exception{

		int batchSize = 64;     
        int nEpochs = numOfEpochs;        
        int truncateReviewsToLength = 256;  
        final int seed = 123;     
        
        Nd4j.getMemoryManager().setAutoGcWindow(10000);  
        
        //Nastavení sítě
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(5e-3))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                
                
                
                .layer(new GravesBidirectionalLSTM.Builder()
                		.nIn(vectorSize)
                		.nOut(256)
                		.dropOut(0.5)
                		.build())
                
                               
                .layer(new DenseLayer.Builder() 
            		    .nIn(256)
            		    .nOut(256)
            		    .activation(Activation.RELU)
            		    .dropOut(0.5)
            		    .build())
                

                
                .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(256)
                        .nOut(2)
                        .build())
                 
                .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            
        //načtení word2Vec modelu
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
        
        //Vytvoření iterátorů
        DataIterator train = new DataIterator(dataFalse, dataTrue, wordVectors, batchSize, truncateReviewsToLength);
        DataIterator test = new DataIterator(dataTestFalse, dataTestTrue, wordVectors, batchSize, truncateReviewsToLength);
        
        //trénovaní
        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(1000), new EvaluativeListener(train, 1, InvocationType.EPOCH_END));
        net.addListeners(new EvaluativeListener(test, 1, InvocationType.EPOCH_END));
        net.fit(train, nEpochs);
		
	}
}
