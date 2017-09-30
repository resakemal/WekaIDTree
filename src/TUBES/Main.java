/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TUBES;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author DXGeneral
 */
public class Main {

    Instances rawData, playData;
    myID3 id3;
    Evaluation E;
        
    
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
        Object[] prediction = new Object[test.numInstances()];
        myID3 cls = new myID3();
        cls.buildClassifier(train);
        System.out.println("check");
        eval.evaluateModel(cls, test, prediction);
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    // save model/hipotesis ke file external
    public void saveModelID3(Classifier C, String dir) throws Exception
    {
        dir += ".model";
        weka.core.SerializationHelper.write(dir, C);
    }
    
    // load model/hipotesis dari file external
    public static Classifier load(String dir) throws Exception {
        return (Classifier) weka.core.SerializationHelper.read(dir);
    }
    
    public void prepareID3(String fileName) throws Exception
    {
        id3 = new myID3();

        DataSource source = new DataSource(fileName);
        rawData = new Instances(source.getDataSet());
        rawData.setClassIndex(rawData.numAttributes() - 1);

        Discretize D = new Discretize();
        D.setInputFormat(rawData);
        playData = Filter.useFilter(rawData, D);
        System.out.println(playData);

        id3.buildClassifier(playData);
        
        System.out.println(id3.toString());
    }    
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        Main model = new Main();
        
        String fileName = "car.data"; // weather.nominal.arff buying,maint,doors,persons,lug_boot,safety
        
        System.out.println("Build ID3 with "+fileName);
        model.prepareID3(fileName); 
        System.out.println();
        
        System.out.println("Full Training - ID3 with "+fileName);
        model.evaluate(model.id3, model.playData);
        
        System.out.println("10-Fold Cross Validation - ID3 with "+fileName);
        model.validate(model.id3, model.playData);
        
        System.out.println("Split Test - ID3 with "+fileName);
        model.split_test(model.id3, model.playData);
        
    }
    
}
