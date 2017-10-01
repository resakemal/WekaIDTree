/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TUBES;

import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;


/**
 *
 * @author DXGeneral
 */
public class Main {

    myID3 id3;
    myC45 c45;
    
    String fileName;
    // membuat instance baru dari masukan pengguna
    public Instance makeNewInstance(Instances data) {
        
        double[] values = new double[data.numAttributes()];
        Scanner in = new Scanner(System.in);

        for(int i = 0; i < data.numAttributes() - 1; i++) {
                System.out.print(data.attribute(i).name() + " : ");
                values[i] = in.nextDouble();
        }

        Instance instance = new Instance(1.0, values);
        // You have to associate every new instance you create to an Instances object using setDataset.
        instance.setDataset(data);
        return instance;
    }
    
    public void evaluate(Classifier classifier, Instances data) throws Exception
    {
        /* Full Training */
        
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier, data);
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void validate(Classifier classifier, Instances data) throws Exception
    {
        /* 10-Fold Cross Validation */
        
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void split_test(Classifier classifier, Instances data) throws Exception
    {
        /* Split Test */
        Scanner scan = new Scanner(System.in);
        
        System.out.print("Masukkan rate split test (dalam %) >>> "); int rate = scan.nextInt();
        
        data.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(data.numInstances() * rate/100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Evaluation eval = new Evaluation(test);
        Classifier cls = classifier;
        cls.buildClassifier(train);
        System.out.println("check");
        eval.evaluateModel(cls, test);
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void test(Classifier classifier, Instances data) throws Exception{
        System.out.println("Evaluasi model dengan Test Set.");
        DataSource testEval;
        Instances testData;
        Instances evalData = null;
        boolean valid = true;
        Scanner scan = new Scanner(System.in);

        try {
            System.out.print("Masukkan nama file test >>> ");
            testEval = new DataSource (scan.next());
            testData = new Instances(testEval.getDataSet());
            testData.setClassIndex(testData.numAttributes() - 1);

            System.out.print("Diskritisasi data? (Y/N) >>> ");
            String isFilter = scan.next();
            System.out.println();

            if (isFilter.equals("y") || isFilter.equals("Y")) {
                Filter discretize = new Discretize();
                discretize.setInputFormat(testData);
                evalData = Filter.useFilter(testData, discretize);
            } else if (isFilter.equals("n") || isFilter.equals("N")) {
                evalData = testData;
            } else {
                valid = false;
                System.out.println("Harap masukkan 'Y' atau 'N'.");
            }

        } catch (Exception ex) {
           
        }

        if (valid) {
            Evaluation evaluation;
            try {

                evaluation = new Evaluation(data);
                evaluation.evaluateModel(classifier, evalData);

                System.out.println(evaluation.toSummaryString());
                System.out.println(evaluation.toClassDetailsString());
                System.out.println(evaluation.toMatrixString());
            } catch (Exception ex) {
                System.out.println("Terjadi error saat pengevaluasian\n");
            }
        }
    }
    
    // save model/hipotesis ke file external
    public void save(Classifier C, String dir) throws Exception
    {
        dir += ".model";
        weka.core.SerializationHelper.write(dir, C);
    }
    
    // load model/hipotesis dari file external
    public Classifier load(String dir) throws Exception {
        return (Classifier) weka.core.SerializationHelper.read(dir);
    }
    
    public void prepareID3(Instances data) throws Exception
    {
        id3 = new myID3();
        
        id3.buildClassifier(data);
        
        System.out.println(id3.toString());
    }

    public void prepareC45(Instances data) throws Exception
    {
        c45 = new myC45();

        c45.buildClassifier(data);
        
        System.out.println(c45.toString());
    }        
    
    public Instances prepareData(String fileName) throws Exception
    {
        Scanner in = new Scanner(System.in);
        
        System.out.print("Use Resampling? (Y/N) >>> ");
        boolean useResample = (in.next().toUpperCase().charAt(0) == 'Y' ? true : false);
        System.out.println();
        
        System.out.print("Use Remove Attribute? (Y/N) >>> ");
        boolean useRemove = (in.next().toUpperCase().charAt(0) == 'Y' ? true : false);
        System.out.println();
        
        DataSource source = new DataSource(fileName);
        Instances data = new Instances(source.getDataSet());
        data.setClassIndex(data.numAttributes() - 1);
        
        if(useResample){
            Resample R = new Resample();
            R.setInputFormat(data);
            data = Filter.useFilter(data, R);
        }
        
        if(useRemove){
            Remove remove = new Remove();
            System.out.print("Pilih Atribut yang ingin dihilangkan (input int) >>> ");
            remove.setAttributeIndices(in.next());
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
        }
        System.out.println(data);
        
        return data;
    }
    
    public static void mainMenu(Instances data, Main model) throws Exception{
        Scanner in = new Scanner(System.in);
        
        int pilihan;
        
        do{
            System.out.println("Pilih algoritma yang akan digunakan");
            System.out.println("1. ID3");
            System.out.println("2. C45");
            pilihan = in.nextInt();
            
            switch (pilihan){
                    
                case 1: runID3(data, model);
                break;

                case 2: runC45(data, model);
                break;
                
                default: break;
             }
            
            
        }while(pilihan != -99);
    }
    
    public static void runID3(Instances data, Main model) throws Exception{

        /* Discretize Data to handle Numeric value */
        Discretize D = new Discretize();
        D.setInputFormat(data);
        data = Filter.useFilter(data, D);
        
        /* BUILD ID3 CLASSIFIER */
        
        System.out.println("BUILD ID3 WITH "+model.fileName);
        model.prepareID3(data); 
        System.out.println();
        
        /* TESTING & EVALUATION */
        
        System.out.println("FULL TRAINING - ID3 with "+model.fileName);
        model.evaluate(model.id3, data);
        
        System.out.println("10-FOLD CROSS VALIDATION - ID3 with "+model.fileName);
        model.validate(model.id3, data);
        
        System.out.println("SPLIT TEST - ID3 with "+model.fileName);
        model.split_test(model.id3, data);
        
        /* MODEL TEST USING DATA SET */
        model.test(model.id3, data);

        /* Classifying Instance */
        
        Instance test = model.makeNewInstance(data);
        System.out.println(data.classAttribute().value((int) model.id3.classifyInstance(test)));

        /* SAVE-LOAD MODEL */
        
        System.out.println("Save ID3");
        model.save(model.id3, "ID3");
        
        myID3 x = (myID3) model.load("ID3.model");
        
        model.evaluate(x, data);
    }
    
    public static void runC45(Instances data, Main model){
         /* BUILD C4.5 CLASSIFIER */
        
        System.out.println("Build C4.5 with "+model.fileName);
        try { 
            model.prepareC45(data);
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println();
        
        /* TESTING & EVALUATION */
        try{
            System.out.println("FULL TRAINING - ID3 with "+model.fileName);
            model.evaluate(model.c45, data);

            System.out.println("10-FOLD CROSS VALIDATION - ID3 with "+model.fileName);
            model.validate(model.c45, data);

            System.out.println("SPLIT TEST - ID3 with "+model.fileName);
            model.split_test(model.c45, data);
        } catch (Exception ex){
            //System.out.println(ex.getMessage());
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }

        try {
            /* MODEL TEST USING DATA SET */
            model.test(model.c45, data);
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        /* Classifying Instance */
        
        Instance test = model.makeNewInstance(data);
        System.out.println(data.classAttribute().value((int) model.c45.classifyInstance(test)));

        /* SAVE-LOAD MODEL */
        
//        System.out.println("Save C45");
//        model.save(model.id3, "C45");
//        
//        myC45 x = (myC45) model.load("C45.model");
        
//        model.evaluate(x, playData);
    }
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        Scanner scan = new Scanner(System.in);
        
        Instances rawData;
        
        Main model = new Main();
        
        /* LOAD DATA */
        
        System.out.print("Masukkan nama file -> "); model.fileName = scan.next();// String fileName = "weather.nominal.arff"
        rawData = model.prepareData(model.fileName);
        
        try{
            mainMenu(rawData, model);
        }catch(Exception x){
            System.out.println(x.getMessage());
        }    
    }
    
}
