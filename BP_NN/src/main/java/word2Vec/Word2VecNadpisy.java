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

public class Word2VecNadpisy {
private static Logger log = LoggerFactory.getLogger(Word2VecModelCreator.class);
    
    //složka, ve které se nachází v příslušných podsložkách všechny textové soubory
    private static final String test = "/users/Marek/Desktop/BP/datanadpisy/all/2zprac";
    
    
    public static void main(String[] args) throws Exception {
    	
    	SentenceIterator iterator = new FileSentenceIterator(new File(test));
        
        SentenceDataPreProcessor.setPreprocessor(iterator);
        final TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new EndingPreProcessor());
        
        final Word2Vec model = new Word2Vec.Builder()
                                        .iterate(iterator)
                                        .tokenizerFactory(tokenizerFactory)
                                        .minWordFrequency(2)
                                        .layerSize(32)
                                        .seed(42)
                                        .epochs(20)
                                        .windowSize(5)
                                        .build();
        
        log.info("Fitting Word2Vec model....");
        model.fit();
        
        /*
        //zkouška modelu
        final Collection<String> words = model.wordsNearest("season",10);
        for(final String word: words){
            System.out.println(word+ " ");
        }
        final double cosSimilarity = model.similarity("season","program");
        System.out.println(cosSimilarity);
        */
        
        // uložení modelu
        WordVectorSerializer.writeWord2VecModel(model, "modelnadpisy32.zip");
}
}