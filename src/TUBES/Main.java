/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TUBES;

import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;


/**
 *
 * @author DXGeneral
 */
public class Main {

    Instances rawData, playData;
    myID3 id3;
//    myC45 c45;
    Evaluation E;
        
    
    // membuat instance baru dari masukan pengguna
    public Instance makeNewInstance() {
        
        double[] values = new double[playData.numAttributes()];
        Scanner in = new Scanner(System.in);

        for(int i = 0; i < playData.numAttributes() - 1; i++) {
                System.out.print(playData.attribute(i).name() + " : ");
                values[i] = in.nextDouble();
        }

        Instance instance = new Instance(1.0, values);
        // You have to associate every new instance you create to an Instances object using setDataset.
        instance.setDataset(playData);
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
        
        data.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Evaluation eval = new Evaluation(test);
        myID3 cls = new myID3();
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
                Instances data1 = this.playData;

                evaluation = new Evaluation(data1);
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

//    public void prepare45(Instances data) throws Exception
//    {
//        c45 = new myC45();
//
//        c45.buildClassifier(data);
//        
//        System.out.println(c45.toString());
//    }        
    
    public void prepareData(String fileName) throws Exception
    {
        boolean useResample = false;
        Scanner in = new Scanner(System.in);
        
        System.out.print("Use Resampling? (Y/N) >>> ");
        useResample = (in.next().toUpperCase().charAt(0) == 'Y' ? true : false);
        System.out.println();
        
        DataSource source = new DataSource(fileName);
        rawData = new Instances(source.getDataSet());
        rawData.setClassIndex(rawData.numAttributes() - 1);
        
        if(useResample){
            Resample R = new Resample();
            R.setInputFormat(rawData);
            rawData = Filter.useFilter(rawData, R);
        }

        Discretize D = new Discretize();
        D.setInputFormat(rawData);
        playData = Filter.useFilter(rawData, D);
        System.out.println(playData);
    }
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        Main model = new Main();
        
        /* LOAD DATA */
        
        String fileName = "car.data"; // weather.nominal.arff
        model.prepareData(fileName);
        
        /* BUILD ID3 CLASSIFIER */
        
        System.out.println("Build ID3 with "+fileName);
        model.prepareID3(model.playData); 
        System.out.println();
        
        /* TESTING & EVALUATION */
        
        System.out.println("Full Training - ID3 with "+fileName);
        model.evaluate(model.id3, model.playData);
        
        System.out.println("10-Fold Cross Validation - ID3 with "+fileName);
        model.validate(model.id3, model.playData);
        
        System.out.println("Split Test - ID3 with "+fileName);
        model.split_test(model.id3, model.playData);
        
        /* MODEL TEST USING DATA SET */
        model.test(model.id3, model.playData);

        /* Classifying Instance */
        
        Instance test = model.makeNewInstance();
        System.out.println(model.playData.classAttribute().value((int) model.id3.classifyInstance(test)));

        /* SAVE-LOAD MODEL */
        
//        System.out.println("Save ID3");
//        model.save(model.id3, "ID3");
        
//        myID3 x = (myID3) model.load("ID3.model");
//        
//        model.evaluate(x, model.playData);

        /* BUILD C4.5 CLASSIFIER */
        
//        System.out.println("Build C4.5 with "+fileName);
//        model.prepareC45(); 
//        System.out.println();
        
        /* TESTING & EVALUATION */
        
//        System.out.println("Full Training - ID3 with "+fileName);
//        model.evaluate(model.c45, model.playData);
//        
//        System.out.println("10-Fold Cross Validation - ID3 with "+fileName);
//        model.validate(model.c45, model.playData);
//        
//        System.out.println("Split Test - ID3 with "+fileName);
//        model.split_test(model.c45, model.playData);

        /* Classifying Instance */
        
//        Instance test = model.makeNewInstance();
//        System.out.println(model.playData.classAttribute().value((int) model.c45.classifyInstance(test)));

        /* SAVE-LOAD MODEL */
        
//        System.out.println("Save C45");
//        model.save(model.id3, "C45");
        
//        myC45 x = (myC45) model.load("C45.model");
//        
//        model.evaluate(x, model.playData);
        
    }
    
}
