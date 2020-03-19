package lstm;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;


/*
 * třída převzatá z příkladů od deeplearning4j a upravená, původní na:
 * https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/SentimentExampleIterator.java
 * 
 * kometáře ponechané v angličtině
 */
public class DataIterator implements DataSetIterator{
	private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    
    private int cursor = 0; 
    
    private final File[] trueNews;
    private final File[] falseNews;
    
    private final TokenizerFactory tokenizerFactory;
    
    public DataIterator(String fal, String tru, WordVectors wordVectors, int batchSize, int truncateLength) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        
    	//cesty k datům
    	File falseFiles = new File(fal);
        File trueFiles = new File(tru);

        falseNews = falseFiles.listFiles();
        trueNews = trueFiles.listFiles();
        
        
        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }
    
    @Override
    public DataSet next(int num) {
        if (cursor >= totalExamples()) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        //First: load reviews to String. Alternate positive and negative reviews
        List<String> reviews = new ArrayList<>(num);
        boolean[] positive = new boolean[num];
        for( int i=0; i<num && cursor<totalExamples(); i++ ){
            if(cursor % 2 == 0){
                //Load positive review
                int posReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(trueNews[posReviewNumber], (Charset)null);
                reviews.add(review);
                positive[i] = true;
            } else {
                //Load negative review
                int negReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(falseNews[negReviewNumber], (Charset)null);
                reviews.add(review);
                positive[i] = false;
            }
            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        int maxLength = 0;
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(new int[]{reviews.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{reviews.size(), 2, maxLength}, 'f');    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);

            // Get the truncated sequence length of document (i)
            int seqLength = Math.min(tokens.size(), maxLength);

            // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
            final INDArray vectors = wordVectors.getWordVectors(tokens.subList(0, seqLength)).transpose();

            // Put wordvectors into features array at the following indices:
            // 1) Document (i)
            // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
            // 3) All elements between 0 and the length of the current sequence
            features.put(
                new INDArrayIndex[] {
                    NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, seqLength)
                },
                vectors);

            // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
            featuresMask.get(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)}).assign(1);

            int idx = (positive[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    public int totalExamples() {
        //return trueNews.length + falseNews.length;
        return Math.min(trueNews.length*2,falseNews.length*2);
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < totalExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
    @Override
    public  DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** Convenience method for loading review to String */
    public String loadReviewToString(int index) throws IOException{
        File f;
        if(index%2 == 0) f = trueNews[index/2];
        else f = falseNews[index/2];
        return FileUtils.readFileToString(f);
    }

    /** Convenience method to get label for review */
    public boolean isPositiveReview(int index){
        return index%2 == 0;
    }

    /**
     * Used post training to load a review from a file to a features INDArray that can be passed to the network output method
     *
     * @param file      File to load the review from
     * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
     * @return          Features array
     * @throws IOException If file cannot be read
     */
    public INDArray loadFeaturesFromFile(File file, int maxLength) throws IOException {
        String review = FileUtils.readFileToString(file, (Charset)null);
        return loadFeaturesFromString(review, maxLength);
    }

    /**
     * Used post training to convert a String to a features INDArray that can be passed to the network output method
     *
     * @param reviewContents Contents of the review to vectorize
     * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
     * @return Features array for the given input String
     */
    public INDArray loadFeaturesFromString(String reviewContents, int maxLength){
        List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
        List<String> tokensFiltered = new ArrayList<>();
        for(String t : tokens ){
            if(wordVectors.hasWord(t)) tokensFiltered.add(t);
        }
        int outputLength = Math.min(maxLength,tokensFiltered.size());

        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        int count = 0;
        for( int j=0; j<tokensFiltered.size() && count<maxLength; j++ ){
            String token = tokensFiltered.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            if(vector == null){
                continue;   //Word not in word vectors
            }
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
            count++;
        }

        return features;
    }
}
