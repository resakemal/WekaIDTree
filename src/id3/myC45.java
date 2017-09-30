/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3;

import java.io.Serializable;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Utils;
import weka.core.NoSupportForMissingValuesException;

import java.util.Enumeration;

public class myC45 extends Classifier implements Serializable{
    
  
  /** Children of this Nodes. Exist if not leaves*/ 
  private myC45[] child_Nodes;

  /** Attribute used for splitting. Null if leaves*/
  private Attribute main_Attribute;

  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution if node is leaf. */
  private double[] class_Distribution;

  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;
  
//    private final Instances dataIris;
    
  /** Default class value if receiving missing value */
    public myC45(){
//        this.dataIris = DataSource.read("G:/STEI/STI/Semester 7/IF 4071 Pembelajaran Mesin/Tubes 1/weka-3-6-14/iris.2D.arff");
        System.out.println("Hello, World!"); 
    }

    private static class Default {
        private Attribute classAttribute;
        private double classValue;
        
        private Default(Attribute classAttribute, double classValue){
            this.classAttribute = classAttribute;
            this.classValue = classValue;
        }
        
        private Attribute classAttribute(){
            
            return this.classAttribute;
        }
        
        private double classValue(){
            
            return this.classValue;
        }
    }
    
    /**
   * Returns default capabilities of the classifier.
   * Function exist in super class Classifier
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  /**
   * Builds C45 decision tree classifier.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  
  @Override
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    Default default_data = computeDefaultValue(data);
    makeTree(data, default_data);
  }

  /**
   * Method for building a C45 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, Default default_data) throws Exception {

    // Check if no instances have reached this node. Missing Value
    if (data.numInstances() == 0) {
      main_Attribute = null;
      m_ClassValue = default_data.classValue();
      m_ClassAttribute = default_data.classAttribute();
      class_Distribution = new double[data.numClasses()];
      return;
    }
    

    // Compute attribute with maximum information gain.
    // If attribute is continuous valued, define discrete values for attribute
    double[] gainRatio = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      gainRatio[att.index()] = computeInfoGain(data, att)/computeSplitInfo(data,att);
    }
    main_Attribute = data.attribute(Utils.maxIndex(gainRatio));
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(gainRatio[main_Attribute.index()], 0)) {
      main_Attribute = null;
      class_Distribution = new double[data.numClasses()];
      Enumeration instEnum = data.enumerateInstances();
      while (instEnum.hasMoreElements()) {
        Instance inst = (Instance) instEnum.nextElement();
        class_Distribution[(int) inst.classValue()]++;
      }
      Utils.normalize(class_Distribution);
      m_ClassValue = Utils.maxIndex(class_Distribution);
      m_ClassAttribute = data.classAttribute();
    } else {
      Instances[] splitData = splitData(data, main_Attribute);
      child_Nodes = new myC45[main_Attribute.numValues()];
      for (int j = 0; j < main_Attribute.numValues(); j++) {
        child_Nodes[j] = new myC45();
        child_Nodes[j].makeTree(splitData[j], default_data);
      }
    }
  }
  
  /**
   * Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  @Override
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (main_Attribute == null) {
      return m_ClassValue;
    } else {
      return child_Nodes[(int) instance.value(main_Attribute)].classifyInstance(instance);
    }
  }

  /**
   * Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  
  @Override
  public double[] distributionForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (main_Attribute == null) {
      return class_Distribution;
    } else { 
      return child_Nodes[(int) instance.value(main_Attribute)].
        distributionForInstance(instance);
    }
  }
  
  /**
   * Computes default value for the tree.
   *
   * @param data the data for which info gain is to be computed
   * @return object data of default class value and attribute
   * @throws Exception if process fails
   */
  private Default computeDefaultValue(Instances data) 
    throws Exception {

        double[] temp_Distribution = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          class_Distribution[(int) inst.classValue()]++;
        }
        Utils.normalize(class_Distribution);
        Default default_leaf = new Default(data.classAttribute(), Utils.maxIndex(class_Distribution));
        
        return default_leaf;
  }

  /**
   * Computes information gain for an attribute.
   *
   * @param data the data for which info gain is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   * @throws Exception if computation fails
   */
  private double computeInfoGain(Instances data, Attribute att) 
    throws Exception {

    double infoGain = computeEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }
  
    /**
   * Computes information gain for an attribute.
   *
   * @param data the data for which info gain is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   * @throws Exception if computation fails
   */
  private double computeSplitInfo(Instances data, Attribute att) throws Exception {
      
    double [] valueCount = new double[att.numValues()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      valueCount[(int) inst.value(att)]++;
    }
    double entropy = 0;
    for (int j = 0; j < att.numValues(); j++) {
      if (valueCount[j] > 0) {
        entropy -= (valueCount[j] / (double) data.numInstances()) * Utils.log2(valueCount[j] / (double) data.numInstances());
      }
    }
    return entropy;
  }

  /**
   * Computes the entropy of a dataset.
   * 
   * @param data the data for which entropy is to be computed
   * @return the entropy of the data's class distribution
   * @throws Exception if computation fails
   */
  private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    for (int j = 0; j < data.numClasses(); j++) {
      if (classCounts[j] > 0) {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) data.numInstances();
    return entropy + Utils.log2(data.numInstances());
  }

  /**
   * Splits a dataset according to the values of a nominal attribute.
   *
   * @param data the data which is to be split
   * @param att the attribute to be used for splitting
   * @return the sets of instances produced by the split
   */
  private Instances[] splitData(Instances data, Attribute att) {

    Instances[] splitData = new Instances[att.numValues()];
    for (int j = 0; j < att.numValues(); j++) {
      splitData[j] = new Instances(data, data.numInstances());
    }
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      splitData[(int) inst.value(att)].add(inst);
    }
      for (Instances splitData1 : splitData) {
          splitData1.compactify();
      }
    return splitData;
  }
}
