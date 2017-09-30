/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TUBES;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author DXGeneral
 */
public class Main {

    Instances rawData, playData;
    myID3 id3;
    Evaluation E;
        
    
    /**
     * @throws java.io.IOException
     */
    
//    public void loadFile() throws IOException, Exception
//    {
//        //ArffLoader loader = new ArffLoader();
//        //loader.setSource(new File("iris.arff"));
//        //Instances in = loader.getDataSet();
//        rawData = DataSource.read("iris.arff");
//        rawData.setClassIndex(rawData.numAttributes() - 1);
//    }
//    
//    public void filter() throws Exception
//    {
//        Discretize D = new Discretize();
//        D.setInputFormat(rawData);
//        playData = Filter.useFilter(rawData, D);
//    }
    
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
    
    public void prepareID3(String fileName) throws Exception
    {
        id3 = new myID3();

        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        rawData = new Instances(reader);
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
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        Main model = new Main();
        
        String fileName = "weather.nominal.arff";
        
        System.out.println("Build ID3 with "+fileName);
        model.prepareID3(fileName); 
        System.out.println();
        
        System.out.println("Evaluate ID3 with "+fileName);
        model.evaluate(model.id3, model.playData);
        
        System.out.println("Validate ID3 with "+fileName);
        model.validate(model.id3, model.playData);
        
    }
    
}
