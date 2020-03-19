package cnn;

import java.io.IOException;
import java.io.PrintStream;



public class LauncherCNN {
	
	//cesty k datům
	public static final String DATA_FALSE_NEWS = "/users/Marek/Desktop/BP/data/labeleddataall/false";
	public static final String DATA_TRUE_NEWS = "/users/Marek/Desktop/BP/data/labeleddataall/true";
	
	public static final String DATA_FALSE_NEWS_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatafiltered/false";
	public static final String DATA_TRUE_NEWS_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatafiltered/true";
	
	public static final String DATA_FALSE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatest/false";
	public static final String DATA_TRUE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatest/true";
	
    public static final String WORD_VECTORS_PATH = "/users/Marek/Desktop/BP/datapokus/GoogleNews-vectors-negative300.bin.gz";
    public static final String WORD_VECTORS_MINE = "/users/Marek/Desktop/BP/datapokus/model.zip";
    
    public static final String NADPISY_FALSE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/train/false";
    public static final String NADPISY_TRUE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/train/true";
    
    public static final String NADPISY_FALSE_TEST = "/users/Marek/Desktop/BP/datanadpisy/test/false";
    public static final String NADPISY_TRUE_TEST = "/users/Marek/Desktop/BP/datanadpisy/test/true";
    
    
    public static void main(String[] args) throws IOException {
		
    	PrintStream fileOut = new PrintStream("./out.txt");
    	System.setOut(fileOut);
    	System.setErr(fileOut);
    	CNNwithWord2Vec pokus = new CNNwithWord2Vec();
    	/*
    	 * jako vstupy jsou zde použity textové soubory, každý textový soubor je jedna zpráva
    	 * do metody posíláme odkaz na složky, ve kterých se nachází jednotlivé soubory
    	 * 
    	 * dále pak již vytvořený word2Vec model a počet dimenzí, které má jeden vektor (300 u google word2Vec modelu)
    	 */
    	pokus.basicsettings(NADPISY_FALSE_TRAIN, NADPISY_TRUE_TRAIN, NADPISY_FALSE_TEST, NADPISY_TRUE_TEST, WORD_VECTORS_PATH, 300, 2);
    	
       }
}
