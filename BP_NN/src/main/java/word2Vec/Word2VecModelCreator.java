package word2Vec;

import java.io.File;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 * Vytvořené podle modelu od deeplearning4j
 * https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java
 * 
 * vytvořené na všech datech (testovacích i trénovacích)
 */
public class Word2VecModelCreator {
    private static Logger log = LoggerFactory.getLogger(Word2VecModelCreator.class);
    
    //složka, ve které se nachází v příslušných podsložkách všechny textové soubory
    private static final String test = "/users/Marek/Desktop/BP/data/all";
    
    
    public static void main(String[] args) throws Exception {
    	
    	SentenceIterator iterator = new FileSentenceIterator(new File(test));
        
        SentenceDataPreProcessor.setPreprocessor(iterator);
        final TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new EndingPreProcessor());

        final Word2Vec model = new Word2Vec.Builder()
                                        .iterate(iterator)
                                        .tokenizerFactory(tokenizerFactory)
                                        .minWordFrequency(5)
                                        .layerSize(100)
                                        .seed(42)
                                        .epochs(10)
                                        .windowSize(5)
                                        .build();
        
        log.info("Fitting Word2Vec model....");
        model.fit();
        
        //zkouška modelu
        final Collection<String> words = model.wordsNearest("season",10);
        for(final String word: words){
            System.out.println(word+ " ");
        }
        final double cosSimilarity = model.similarity("season","program");
        System.out.println(cosSimilarity);
        
        // uložení modelu
        WordVectorSerializer.writeWord2VecModel(model, "model.zip");
}}
    
