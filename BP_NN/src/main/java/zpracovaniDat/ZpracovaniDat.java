package zpracovaniDat;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.io.FileUtils;
import com.opencsv.CSVReader;


	/*
	 * Tato třída vytvoří z dat, která jsou v csv souboru textové soubory, kde každý soubor reprezentuje jednu zprávu  
	 * název nového souboru reprezentuje id zprávy, obsahem jsou pak bud nadpisy či texty jenotlivých zpráv
	 */
public class ZpracovaniDat {
	
	//trénovací data formát - id,title,author,text,label
	private static final String soubortrain = "/users/Marek/Desktop/BP/train.csv";
	
	//testovací data jsou ve dvou souborech
	//první soubor formát - id,title,author,text
	private static final String soubortest = "/users/Marek/Desktop/BP/test.csv";
	//druhý soubor formát - id, label
	private static final String souborlabely = "/users/Marek/Desktop/BP/submit.csv";
	
	List<List<String>> records = new ArrayList<>();
	
	// data kratší než daný počet znaků budou vypsána na konzoli a vymazána
	private int delkaTextu;
	public static void main(String[] args) throws Exception{
		
    	ZpracovaniDat hola = new ZpracovaniDat();
		hola.TestovaciDataTexty(100);
		
	}
	
	
	//metoda pro vytvoření trénovacích dat dat
	@SuppressWarnings("deprecation")
	private void TrenovaciDataTexty (int delkaTextu) throws IOException, InterruptedException {
		int chybiText = 0;
		int chybiLabel = 0;
		int kratkyText = 0;
		List<String> texty = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		Reader reader = Files.newBufferedReader(Paths.get(soubortrain));
		CSVReader csvReader = new CSVReader(reader);

		{
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                texty.add(nextRecord[3]);
                labels.add(nextRecord[4]);
            }
        }
		
		
		for(int i=1; i< labels.size(); i++) {
			String veta = texty.get(i);
			String upravenaVeta = veta.replaceAll("[^a-zA-Z0-9\\s]", "");
			texty.set(i, upravenaVeta);
		}
		
		/*
		 * analýza zda některá pole v datech nechybí či nějsou příliš krátká
		 */
		
		for(int i=0; i<texty.size(); i++) {
			if(texty.get(i).isEmpty()) {
				System.out.println("chybí text v souboru s id " + i);
				chybiText++;
			}
			
			if(labels.get(i).isEmpty()) {
				System.out.println("chybí label v souboru s id " + i);
				chybiLabel++;
			}
			
			if(texty.get(i).length() < delkaTextu) {
				System.out.println("text je kratší než " +delkaTextu+ " znaků " + i );
				System.out.println(texty.get(i));
				kratkyText++;
			}
			
		}
		
		System.out.println("chybí text v "+ chybiText + " případech");
		System.out.println("chybí label v "+ chybiLabel + " případech");
		System.out.println("text kratší než "+ delkaTextu + " znaků v " + kratkyText + " případech");
		

		
		for(int i=1; i< labels.size(); i++) {
			if(Integer.parseInt(labels.get(i)) == 0 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/data/labeleddatafiltered/falsetrain", String.valueOf(i)), texty.get(i));
			}
			
			if(Integer.parseInt(labels.get(i)) == 1 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/data/labeleddatafiltered/truetrain", String.valueOf(i)), texty.get(i));
			}
			
			}
		
		
	}
	
	@SuppressWarnings("deprecation")
	private void TestovaciDataTexty (int delkaTextu) throws IOException, InterruptedException {
		int chybiText = 0;
		int chybiLabel = 0;
		int kratkyText = 0;
		int truetext = 0;
		int falsetext = 0;
		
		List<String> texty = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		Reader reader = Files.newBufferedReader(Paths.get(soubortest));
		Reader reader2 = Files.newBufferedReader(Paths.get(souborlabely));
		CSVReader csvReader = new CSVReader(reader);
		CSVReader csvReader2 = new CSVReader(reader2);

		{
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                texty.add(nextRecord[3]);
            }
        }
		
		{

            String[] nextRecord;
            while ((nextRecord = csvReader2.readNext()) != null) {
                labels.add(nextRecord[1]);
            }
        }
		
		
		for(int i=1; i< labels.size(); i++) {
			String veta = texty.get(i);
			String upravenaVeta = veta.replaceAll("[^a-zA-Z0-9\\s]", "");
			texty.set(i, upravenaVeta);
		}
		
		/*
		 * analýza zda některá pole v datech nechybí či nějsou příliš krátká
		 */
		
		for(int i=0; i<texty.size(); i++) {
			if(texty.get(i).isEmpty()) {
				System.out.println("chybí text v souboru s id " + i);
				chybiText++;
			}
			
			if(labels.get(i).isEmpty()) {
				System.out.println("chybí label v souboru s id " + i);
				chybiLabel++;
			}
			
			if(texty.get(i).length() < delkaTextu) {
				System.out.println("text je kratší než " +delkaTextu+ " znaků " + i );
				System.out.println(texty.get(i));
				kratkyText++;
			}
			
		}
		
		System.out.println("chybí text v "+ chybiText + " případech");
		System.out.println("chybí label v "+ chybiLabel + " případech");
		System.out.println("text kratší než "+ delkaTextu + " znaků v " + kratkyText + " případech");
				

		
		for(int i=1; i< labels.size(); i++) {
			if(Integer.parseInt(labels.get(i)) == 0 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/data/labeleddatatestfiltered/falsetest", String.valueOf(i)), texty.get(i));
			falsetext++;
			}
			
			if(Integer.parseInt(labels.get(i)) == 1 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/data/labeleddatatestfiltered/truetest", String.valueOf(i)), texty.get(i));
			truetext++;
			}
			
			}
		
		System.out.println(" počet true " + truetext);
		System.out.println(" počet false " + falsetext);
		
		
	}
	
	@SuppressWarnings("deprecation")
	private void TrenovaciDataNadpisy (int delkaTextu) throws IOException, InterruptedException {
		int chybiText = 0;
		int chybiLabel = 0;
		int kratkyText = 0;
		int truetext = 0;
		int falsetext = 0;
		List<String> texty = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		Reader reader = Files.newBufferedReader(Paths.get(soubortrain));
		CSVReader csvReader = new CSVReader(reader);

		{
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                texty.add(nextRecord[1]);
                labels.add(nextRecord[4]);
            }
        }
		
		
		for(int i=1; i< labels.size(); i++) {
			String veta = texty.get(i);
			String upravenaVeta = veta.replaceAll("[^a-zA-Z0-9\\s]", "");
			texty.set(i, upravenaVeta);
		}
		
		
		for(int i=0; i<texty.size(); i++) {
			if(texty.get(i).isEmpty()) {
				System.out.println("chybí text v souboru s id " + i);
				chybiText++;
			}
			
			if(labels.get(i).isEmpty()) {
				System.out.println("chybí label v souboru s id " + i);
				chybiLabel++;
			}
			
			if(texty.get(i).length() < delkaTextu) {
				System.out.println("text je kratší než " +delkaTextu+ " znaků " + i );
				System.out.println(texty.get(i));
				kratkyText++;
			}
			
		}
		
		System.out.println("chybí text v "+ chybiText + " případech");
		System.out.println("chybí label v "+ chybiLabel + " případech");
		System.out.println("text kratší než "+ delkaTextu + " znaků v " + kratkyText + " případech");
		
		

		
		
		for(int i=1; i< labels.size(); i++) {
			if(Integer.parseInt(labels.get(i)) == 0 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/datanadpisy/all/false2", String.valueOf(i)), texty.get(i));
			falsetext++;
			}
			
			if(Integer.parseInt(labels.get(i)) == 1 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/datanadpisy/all/true2", String.valueOf(i)), texty.get(i));
			truetext++;
			}
			
			}
		
		System.out.println(" počet true " + truetext);
		System.out.println(" počet false " + falsetext);
		
		
	}
	
	@SuppressWarnings("deprecation")
	private void TestovaciDataNadpisy (int delkaTextu) throws IOException, InterruptedException {
		int chybiText = 0;
		int chybiLabel = 0;
		int kratkyText = 0;
		int truetext = 0;
		int falsetext = 0;
		
		List<String> texty = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		Reader reader = Files.newBufferedReader(Paths.get(soubortest));
		Reader reader2 = Files.newBufferedReader(Paths.get(souborlabely));
		CSVReader csvReader = new CSVReader(reader);
		CSVReader csvReader2 = new CSVReader(reader2);

		{
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                texty.add(nextRecord[1]);
            }
        }
		
		
		{

            String[] nextRecord;
            while ((nextRecord = csvReader2.readNext()) != null) {
                labels.add(nextRecord[1]);
            }
        }
		
		
		for(int i=1; i< labels.size(); i++) {
			String veta = texty.get(i);
			String upravenaVeta = veta.replaceAll("[^a-zA-Z0-9\\s]", "");
			texty.set(i, upravenaVeta);
		}
		
		/*
		 * analýza zda některá pole v datech nechybí či nějsou příliš krátká
		 */
		for(int i=0; i<texty.size(); i++) {
			if(texty.get(i).isEmpty()) {
				System.out.println("chybí text v souboru s id " + i);
				chybiText++;
			}
			
			if(labels.get(i).isEmpty()) {
				System.out.println("chybí label v souboru s id " + i);
				chybiLabel++;
			}
			
			if(texty.get(i).length() < delkaTextu) {
				System.out.println("text je kratší než " +delkaTextu+ " znaků " + i );
				System.out.println(texty.get(i));
				kratkyText++;
			}
			
		}
		
		System.out.println("chybí text v "+ chybiText + " případech");
		System.out.println("chybí label v "+ chybiLabel + " případech");
		System.out.println("text kratší než "+ delkaTextu + " znaků v " + kratkyText + " případech");
				

		
		for(int i=1; i< labels.size(); i++) {
			if(Integer.parseInt(labels.get(i)) == 0 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/datanadpisy/all/falsetest2", String.valueOf(i)), texty.get(i));
			falsetext++;
			}
			
			if(Integer.parseInt(labels.get(i)) == 1 && texty.get(i).length() > delkaTextu) {
			FileUtils.writeStringToFile(new File("/users/Marek/Desktop/BP/datanadpisy/all/truetest2", String.valueOf(i)), texty.get(i));
			truetext++;
			}
			
			}
		
		System.out.println(" počet true " + truetext);
		System.out.println(" počet false " + falsetext);
		
		
	}
}
