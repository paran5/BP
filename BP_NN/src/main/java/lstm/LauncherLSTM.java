package lstm;

import java.io.IOException;
import java.io.PrintStream;



public class LauncherLSTM {
	
	public static final String DATA_FALSE_TRAIN = "/users/Marek/Desktop/BP/data/labeleddatafiltered/falsetrain";
	public static final String DATA_TRUE_TRAIN = "/users/Marek/Desktop/BP/data/labeleddatafiltered/truetrain";
	
	public static final String DATA_FALSE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatestfiltered/falsetest";
	public static final String DATA_TRUE_TEST = "/users/Marek/Desktop/BP/data/labeleddatatestfiltered/truetest";

    public static final String WORD_VECTORS_PATH = "/users/Marek/Desktop/BP/datapokus/GoogleNews-vectors-negative300.bin.gz";
    
    public static final String WORD_VECTORS_MINE_NADPISY1 = "/users/Marek/Desktop/BP/datapokus/modelnadpisy1.zip";
    public static final String WORD_VECTORS_MINE_NADPISY32 = "/users/Marek/Desktop/BP/datapokus/modelnadpisy32.zip";   
    
    public static final String WORD_VECTORS_TEXTY1 = "/users/Marek/Desktop/BP/datapokus/model1.zip";
    public static final String WORD_VECTORS_TEXTY100 = "/users/Marek/Desktop/BP/datapokus/model100.zip";    
  
    public static final String NADPISY_FALSE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/all/false2";
    public static final String NADPISY_TRUE_TRAIN = "/users/Marek/Desktop/BP/datanadpisy/all/true2";
    
    public static final String NADPISY_FALSE_TEST = "/users/Marek/Desktop/BP/datanadpisy/all/falsetest2";
    public static final String NADPISY_TRUE_TEST = "/users/Marek/Desktop/BP/datanadpisy/all/truetest2";
    
    
    public static void main(String[] args) throws IOException {
		/*
    	PrintStream fileOut = new PrintStream("./out.txt");
    	System.setOut(fileOut);
    	System.setErr(fileOut);
    	*/
    	
    	LSTM NN = new LSTM();
    	
    	/*
    	 * jako vstupy jsou zde použity textové soubory, každý textový soubor je jedna zpráva
    	 * do metody posíláme odkaz na složky, ve kterých se nachází jednotlivé soubory
    	 * 
    	 * dále pak již vytvořený word2Vec model a počet dimenzí, které má jeden vektor (300 u google word2Vec modelu)
    	 */
    	try {
			NN.basicsettings( DATA_FALSE_TRAIN, DATA_TRUE_TRAIN, DATA_FALSE_TEST, DATA_TRUE_TEST, WORD_VECTORS_TEXTY100, 100, 4);
		} catch (Exception e) {
			e.printStackTrace();
		}
    	
    	
       }
}
