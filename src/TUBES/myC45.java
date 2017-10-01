/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TUBES;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.HashMap;

public class myC45 extends Classifier implements Serializable{
    
  
  /** Children of this Nodes. Exist if not leaves*/ 
  private myC45[] child_Nodes;

  /** Attribute used for splitting. Null if leaves*/
  private Attribute main_Attribute;
  
  /** Used attribute in previous nodes of the tree*/
  private static ArrayList<Attribute> usedAttributes;
  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution of node. */
  private double[] class_Distribution;

  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;
  
  /** Container for storing chosen numeric value 
   * for numeric attributes. */
  private double[] numericValues;
  
  /** Class for calculating data splits. */
  private SplitData splitController;
  
//    private final Instances dataIris;
    
  /** Default class value if receiving missing value */
    public myC45(){
            splitController = new SplitData();
//        this.dataIris = DataSource.read("G:/STEI/STI/Semester 7/IF 4071 Pembelajaran Mesin/Tubes 1/weka-3-6-14/iris.2D.arff");;
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
    
    public Attribute mainAttribute(){
        return this.main_Attribute;
    }
    
    public myC45[] child_Nodes(){
        return this.child_Nodes;
    }
    
    public double classValue(){
        return this.m_ClassValue;
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
    
    Default default_data = computeDefaultValue(data);
    makeTree(data, default_data, null);
  }

  /**
   * Method for building a C45 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, Default default_data, ArrayList<Attribute> used) throws Exception {
    // Initialize variables
    usedAttributes = new ArrayList<>();
    if (used != null){
        usedAttributes = used;
        for (int i = 0; i < usedAttributes.size(); i++) {
            data.deleteAttributeAt(usedAttributes.get(i).index());
        }
    }
    numericValues = new double[data.numAttributes()];
    //System.out.println(data);
    
    // Check if no instances have reached this node. Missing Value
    if (data.numInstances() == 0) {
      main_Attribute = null;
      m_ClassValue = default_data.classValue();
      m_ClassAttribute = default_data.classAttribute();
      class_Distribution = new double[data.numClasses()];
      return;
    }
    
    //Make leaf if all attributes has been used, use most common class
    if (usedAttributes.size() == data.numAttributes()) {
      class_Distribution = new double[data.numClasses()];
      Enumeration classEnum = data.enumerateAttributes();
      while (classEnum.hasMoreElements()) {
         Instance inst = (Instance) classEnum.nextElement(); 
         class_Distribution[(int) inst.classIndex()]++;
      }  
      double chosenClass = Utils.maxIndex(class_Distribution);
      main_Attribute = null;
      m_ClassValue = chosenClass;
      m_ClassAttribute = data.classAttribute();
      return;  
    }
        
    // Compute attribute with maximum information gain.
    // If attribute is continuous valued, define discrete values for attribute
    double[] infoGains = new double[data.numAttributes()];
    double[] gainRatios = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      if (att.isNumeric()) {
        infoGains[att.index()] = calculateNumeric(data, att);
        gainRatios[att.index()] = infoGains[att.index()] / 
            computeSplitInfoNumeric(data, att, numericValues[att.index()]);
      } else {
        infoGains[att.index()] = computeInfoGain(data, att);
        gainRatios[att.index()] = infoGains[att.index()] / computeSplitInfo(data,att);
      }
    }
    main_Attribute = data.attribute(Utils.maxIndex(gainRatios));
    usedAttributes.add(main_Attribute);
    //for (int i = 0; i < numericValues.length; i++)
    //System.out.println(gainRatios[i]);
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(infoGains[main_Attribute.index()], 0)) {
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
      if (!main_Attribute.isNumeric()) {
        child_Nodes = new myC45[main_Attribute.numValues()];
        splitController.splitData(data, main_Attribute);
        splitController.handleMissingValues(data, main_Attribute);
      } else {
        child_Nodes = new myC45[2];
        splitController.splitDataNumeric(data, main_Attribute, numericValues[main_Attribute.index()]);
        splitController.handleMissingValuesNumeric(data, main_Attribute, numericValues[main_Attribute.index()]);
      }
      
      Instances[] splitData = splitController.getSplit();
      //System.out.println(splitData);
      for (int j = 0; j < child_Nodes.length; j++) {
        child_Nodes[j] = new myC45();
        child_Nodes[j].makeTree(splitData[j], default_data, used);
      }
    }
  }
  
  /**
   * Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   */
  @Override
  public double classifyInstance(Instance instance) {

    if (main_Attribute == null) {
      return m_ClassValue;
    } else {
      if (main_Attribute.isNumeric()) {
        if (instance.isMissing(main_Attribute)) {
           return child_Nodes[splitController.getMostCommon()].classifyInstance(instance);
        } else if (instance.value(main_Attribute) <= numericValues[main_Attribute.index()]) {
           return child_Nodes[0].classifyInstance(instance); 
        } else {
           return child_Nodes[1].classifyInstance(instance); 
        }
      } else {
            return child_Nodes[(int) instance.value(main_Attribute)].
        classifyInstance(instance);
      }
    }
  }

  /**
   * Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   */
  
  @Override
  public double[] distributionForInstance(Instance instance) {

    if (main_Attribute == null) {
      return class_Distribution;
    } else {
      if (main_Attribute.isNumeric()) {
        if (instance.isMissing(main_Attribute)) {
           return child_Nodes[splitController.getMostCommon()].distributionForInstance(instance);
        } else if (instance.value(main_Attribute) <= numericValues[main_Attribute.index()]) {
           return child_Nodes[0].distributionForInstance(instance); 
        } else {
           return child_Nodes[1].distributionForInstance(instance); 
        }
      } else {
            return child_Nodes[(int) instance.value(main_Attribute)].
        distributionForInstance(instance);
      }
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
          temp_Distribution[(int) inst.classValue()]++;
        }
        Utils.normalize(temp_Distribution);
        Default default_leaf = new Default(data.classAttribute(), Utils.maxIndex(temp_Distribution));
        
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
    SplitData splitSub = new SplitData();
    splitSub.splitData(data, att);
    Instances[] splitData = splitSub.getSplit();
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
  private double computeNumericInfoGain(Instances data, Attribute att, double value) 
    throws Exception {

    double infoGain = computeEntropy(data);
    SplitData splitSub = new SplitData();
    splitSub.splitDataNumeric(data, att, value );
    Instances[] splitData = splitSub.getSplit();
    for (int j = 0; j < 2; j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }
  
    /**
   * Computes split information for an attribute
   *
   * @param data the data for which split info is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   * @throws Exception if computation fails
   */
  private double computeSplitInfo(Instances data, Attribute att) throws Exception {
      
    double [] valueCount = new double[att.numValues()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      if (!inst.isMissing(att)) {
        valueCount[(int) inst.value(att)]++;
      }
    }
    double splitInfo = 0;
    for (int j = 0; j < att.numValues(); j++) {
      if (valueCount[j] > 0) {
        splitInfo -= (valueCount[j] / (double) data.numInstances()) * Utils.log2(valueCount[j] / (double) data.numInstances());
      }
    }
    return splitInfo;
  }
  
    /**
   * Computes split information for a numeric attribute.
   *
   * @param data the data for which split info is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   * @throws Exception if computation fails
   */
  private double computeSplitInfoNumeric(Instances data, Attribute att, double value) throws Exception {
      
    double [] valueCount = new double[2];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      if (!inst.isMissing(att)) {
        if (inst.value(att) <= value) {
            valueCount[0]++;
        } else {
            valueCount[1]++;
        }
      }
    }
    double splitInfo = 0;
    for (int j = 0; j < 2; j++) {
      if (valueCount[j] > 0) {
        splitInfo -= (valueCount[j] / (double) data.numInstances()) * Utils.log2(valueCount[j] / (double) data.numInstances());
      }
    }
    return splitInfo;
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
   * Select all possible split from data, then randomly choose 10 splits to 
   * compare by gain ratio and select value of best split
   *
   * @param data the data which value of split to be decided
   * @param att the attribute to be used for splitting
   * @return the gain ratio of the best split value
   */
  private double calculateNumeric (Instances data, Attribute att) 
      throws Exception {
      // Sort data for iteration
      data.sort(att);
      
      // Determine splits by class difference in 2 instances
      ArrayList<Double> splitList = new ArrayList<>(); 
      Instance check = data.instance(0);
      Enumeration instEnum = data.enumerateInstances();
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
      if (randomSplit.size() > 0) {
        double[] infoGains = new double[randomSplit.size()];
        for (int i = 0; i < randomSplit.size(); i++) {       
          infoGains[i] = computeNumericInfoGain(data, att, randomSplit.get(i));
        }
        int chosenIndex = Utils.maxIndex(infoGains);
        numericValues[att.index()] = randomSplit.get(chosenIndex);
        return infoGains[chosenIndex];
      } else {
          return 0;
      }
  }
  
    private ArrayList<Double> getChildClasses(myC45 subTree) {
        // If node is not leaf, get child's class value
        ArrayList<Double> childValues = new ArrayList<>();
        if (subTree.main_Attribute != null) {
            for (int i = 0; i < subTree.child_Nodes.length; i++) {
                childValues.addAll(getChildClasses(subTree.child_Nodes[i]));
            }
        } else {
            childValues.add(subTree.m_ClassValue);
        }
        return childValues;
    }
    
    private Double getMajorityClass(myC45 subTree) {
        ArrayList<Double> childValues = getChildClasses(subTree);
        HashMap<Double,Integer> hm = new HashMap<Double,Integer>();
        int max  = 1;
        Double temp = 0.0;

        for(int i = 0; i < childValues.size(); i++) {

            if (hm.get(childValues.get(i)) != null) {

                int count = hm.get(childValues.get(i));
                count++;
                hm.put(childValues.get(i), count);

                if(count > max) {
                    max  = count;
                    temp = childValues.get(i);
                }
            }
            else 
                hm.put(childValues.get(i),1);
        }
        return temp;
    }
    
    private boolean isPrune(myC45 treeNow) {
        return true;
    }

    public void pruneTree() {
//        get tree root
//        change subtree below with leaf or majority
//        measure accuracy
//        if ok then prune
//        else to next non leaf node
    }
  
  /**
   * Prints the decision tree using the private toString method from below.
   *
   * @return a textual description of the classifier
   */
  @Override
  public String toString() {

    if ((class_Distribution == null) && (child_Nodes == null)) {
      return "Id3: No model built yet.";
    }
    return "Id3\n\n" + toString(0);
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
