

package TUBES;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Utils;
import weka.core.FastVector;

import java.util.Enumeration;
import java.util.Random;

public class myC45v2 extends Classifier implements Serializable{
    
  
  /** Children of this Nodes. Exist if not leaves*/ 
  private myC45v2[] child_Nodes;

  /** Attribute used for splitting. Null if leaves*/
  private Attribute main_Attribute;

  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution if node is leaf. */
  private double[] class_Distribution;

  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;
  
  /** Dataset of current node. */
  private Instances dataset;
  
//    private final Instances dataIris;
    
  /** Default class value if receiving missing value */
    public myC45v2(){
       
    }

    private static class Default {
        private final Attribute classAttribute;
        private final double classValue;
        
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
    result.enable(Capability.NUMERIC_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  /**
   * Builds Id3 decision tree classifier.
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
    
    makeTree(data, computeDefaultValue(data));
  }

  /**
   * Method for building an Id3 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, Default default_data) throws Exception {
    dataset = data;

    // Check if no instances have reached this node. Missing Value
    if (dataset.numInstances() == 0) {
      main_Attribute = null;
      m_ClassValue = default_data.classValue();
      m_ClassAttribute = default_data.classAttribute();
      class_Distribution = new double[dataset.numClasses()];
      return;
    }
    
    // Discretize numeric attributes
    Enumeration discEnum = dataset.enumerateAttributes();
    while (discEnum.hasMoreElements()) {
      Attribute att = (Attribute) discEnum.nextElement();
      discretize(dataset, att);
    }

    // Compute attribute with maximum information gain.
    double[] infoGains = new double[dataset.numAttributes()];
    double[] gainRatios = new double[dataset.numAttributes()];
    Enumeration attEnum = dataset.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()] = computeInfoGain(dataset, att);
    }
    main_Attribute = dataset.attribute(Utils.maxIndex(infoGains));
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(infoGains[main_Attribute.index()], 0)) {
      main_Attribute = null;
      class_Distribution = new double[dataset.numClasses()];
      Enumeration instEnum = dataset.enumerateInstances();
      while (instEnum.hasMoreElements()) {
        Instance inst = (Instance) instEnum.nextElement();
        class_Distribution[(int) inst.classValue()]++;
      }
      Utils.normalize(class_Distribution);
      m_ClassValue = Utils.maxIndex(class_Distribution);
      m_ClassAttribute = dataset.classAttribute();
    } else {
      Instances[] splitData = splitData(data, main_Attribute);
      child_Nodes = new myC45v2[main_Attribute.numValues()];
      for (int j = 0; j < main_Attribute.numValues(); j++) {
        child_Nodes[j] = new myC45v2();
        child_Nodes[j].makeTree(splitData[j], computeDefaultValue(data));
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
  public double classifyInstance(Instance instance) {

    if (main_Attribute == null) {
      return m_ClassValue; // m_ClassAttribute.value((int) m_ClassValue) m_ClassValue
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
  public double[] distributionForInstance(Instance instance) {

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

        double[] temp_Distribution = new double[dataset.numClasses()];
        Enumeration instEnum = dataset.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          temp_Distribution[(int) inst.classValue()]++;
        }
        Utils.normalize(temp_Distribution);
        Default default_leaf = new Default(dataset.classAttribute(), Utils.maxIndex(temp_Distribution));
        
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
                     (double) dataset.numInstances()) *
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
  private double computeNumericInfoGain(Instances data, Attribute att, double value) 
    throws Exception {

    double infoGain = computeEntropy(data);
    Instances[] splitData = splitDataNumeric(data, att, value);
    for (int j = 0; j < 2; j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) dataset.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }

  /**
   * Computes the entropy of a dataset.
   * 
   * @param data the data for which entropy is to be computed
   * @return the entropy of the data's class distribution
   * @throws Exception if computation fails
   */
  private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[dataset.numClasses()];
    Enumeration instEnum = dataset.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    for (int j = 0; j < dataset.numClasses(); j++) {
      if (classCounts[j] > 0) {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) dataset.numInstances();
    return entropy + Utils.log2(dataset.numInstances());
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
      splitData[j] = new Instances(data, dataset.numInstances());
    }
    Enumeration instEnum = dataset.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      splitData[(int) inst.value(att)].add(inst);
    }
      for (Instances splitData1 : splitData) {
          splitData1.compactify();
      }
    return splitData;
  }
  
      /**
   * Splits a dataset according to a numeric value into 2 boolean instances
   *
   * @param data the data which is to be split
   * @param att the attribute to be used for splitting
   * @param value the value of numeric attribute for split
   * @return instances array that has been split according to value
   */
    public Instances[] splitDataNumeric(Instances data, Attribute att, double value) {
        
        Instances[] split = new Instances[2];
        for (int j = 0; j < 2; j++) {
          split[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if (!inst.isMissing(att)) {
                if(inst.value(att) <= value) {
                    split[0].add(inst);
                } else {
                    split[1].add(inst);
                }
            }
        }
        
        for (Instances splitData1 : split) {
            splitData1.compactify();
        }
        return split;
    }
  
  /**
   * Discretize numeric attribute by choosing 10 (or less) random candidates
   * and choose one with greatest information gain then swap previous attribute 
   * with new attribute in dataset.
   *
   * @param data the data which discrete value of attribute is to be
   * determined
   * @param att the attribute to be used for splitting
   */
  private void discretize (Instances data, Attribute att) throws Exception {
        // Sort data for iteration
        dataset.sort(att);
      
        // Determine splits by class difference in 2 instances
        ArrayList<Double> splitList = new ArrayList<>(); 
        Instance check = dataset.instance(0);
        Enumeration instEnum = dataset.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          if (!inst.isMissing(att)) {
              if (check.classValue() != inst.classValue()) {
                  splitList.add((check.value(att) + inst.value(att)) / 2);
              }
              check = inst;
          }
        }

        // If there are more than 10 splits, randomly choose 10
        Random rn = new Random();
        ArrayList<Double> randomSplit = new ArrayList<>();
        if (splitList.size() <= 10) {
            randomSplit = splitList;
        } else {
          for (int i = 0; i < 10; i++) {
              randomSplit.add(splitList.get(rn.nextInt(11)));
          }
        }
      
      // Calculate gain ratio for split candidate
        double[] infoGains = new double[randomSplit.size()];
        for (int i = 0; i < randomSplit.size(); i++) {       
          infoGains[i] = computeNumericInfoGain(data, att, randomSplit.get(i));
        }
        double chosenValue = randomSplit.get(Utils.maxIndex(infoGains));
        FastVector attVal = new FastVector();
        attVal.addElement("No");
        attVal.addElement("Yes");
        Attribute newAtt = new Attribute(Double.toString(chosenValue),attVal);
        dataset.insertAttributeAt(newAtt, att.index());
      
        Enumeration instEnum2 = dataset.enumerateInstances();
        double value = Double.valueOf(newAtt.name());
        while (instEnum2.hasMoreElements()) {
            Instance inst = (Instance) instEnum2.nextElement();
            if (!inst.isMissing(att)) {
                if (inst.value(att) <= value)
                    inst.setValue(newAtt, "No");
                else
                    inst.setValue(newAtt, "Yes");
            }
            
        }
        dataset.deleteAttributeAt(att.index());
  }
  
   /**
   * Prints the decision tree using the private toString method from below.
   *
   * @return a textual description of the classifier
   */
  @Override
  public String toString() {

    if ((class_Distribution == null) && (child_Nodes == null)) {
      return "C45: No model built yet.";
    }
    return "C45\n\n" + toString(0);
  }
  
  /**
   * Outputs a tree at a certain level.
   *
   * @param level the level at which the tree is to be printed
   * @return the tree as string at the given level
   */
  private String toString(int level) {

    StringBuilder text = new StringBuilder();
    
    if (main_Attribute == null) {
      if (Instance.isMissingValue(m_ClassValue)) {
        text = text.append(": null");
      } else {
        text = text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));
      } 
    } else {
      for (int j = 0; j < main_Attribute.numValues(); j++) {
        text = text.append("\n");
        for (int i = 0; i < level; i++) {
          text = text.append("|  ");
        }
        text = text.append(main_Attribute.name()).append(" = ").append(main_Attribute.value(j));
        text = text.append(child_Nodes[j].toString(level + 1));
      }
    }
    return text.toString();

  }
}
