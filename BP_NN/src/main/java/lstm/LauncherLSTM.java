package lstm;

import java.io.IOException;
import java.io.PrintStream;



public class LauncherLSTM {
	
	public static final String DATA_FALSE_NEWS = "/users/Marek/Desktop/BP/data/labeleddataall/false";
	public static final String DATA_TRUE_NEWS = "/users/Marek/Desktop/BP/data/labeleddataall/true";
	
	public static final String DATA_FALSE_NEWS_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatafiltered/false";
	public static final String DATA_TRUE_NEWS_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatafiltered/true";
	
	public static final String DATA_FALSE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatest/false";
	public static final String DATA_TRUE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatest/true";
	
	public static final String DATA_FALSE_TEST_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatatestfiltered/false";
	public static final String DATA_TRUE_TEST_FILTERED = "/users/Marek/Desktop/BP/data/labeleddatatestfiltered/true";
	
    public static final String WORD_VECTORS_PATH = "/users/Marek/Desktop/BP/datapokus/GoogleNews-vectors-negative300.bin.gz";
    public static final String WORD_VECTORS_MINE = "/users/Marek/Desktop/BP/datapokus/model.zip";
    
    public static final String NADPISY_FALSE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/train/false";
    public static final String NADPISY_TRUE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/train/true";
    
    public static final String NADPISY_FALSE_TEST = "/users/Marek/Desktop/BP/datanadpisy/test/false";
    public static final String NADPISY_TRUE_TEST = "/users/Marek/Desktop/BP/datanadpisy/test/true";
    
    public static void main(String[] args) throws IOException {
		/*
    	PrintStream fileOut = new PrintStream("./out.txt");
    	System.setOut(fileOut);
    	System.setErr(fileOut);
    	*/
    	
    	LSTMwithWord2Vec pokus = new LSTMwithWord2Vec();
    	
    	/*
    	 * jako vstupy jsou zde použity textové soubory, každý textový soubor je jedna zpráva
    	 * do metody posíláme odkaz na složky, ve kterých se nachází jednotlivé soubory
    	 * 
    	 * dále pak již vytvořený word2Vec model a počet dimenzí, které má jeden vektor (300 u google word2Vec modelu)
    	 */
    	try {
			pokus.basicsettings( DATA_FALSE_NEWS_FILTERED, DATA_TRUE_NEWS_FILTERED, DATA_FALSE_TEST_FILTERED, DATA_TRUE_TEST_FILTERED, WORD_VECTORS_PATH, 300, 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
    	
       }
}
