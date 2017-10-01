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
import weka.classifiers.Evaluation;

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
  
  /** Dataset of current node. */
  private Instances dataset;
  
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
    
    private void copy (myC45 temp) {
        this.child_Nodes = temp.child_Nodes;
        this.class_Distribution = temp.class_Distribution;
        this.dataset = temp.dataset;
        this.main_Attribute = temp.main_Attribute;
        this.m_ClassValue = temp.m_ClassValue;
        this.m_ClassAttribute = temp.m_ClassAttribute;
//        this.numericValues = temp.numericValues;
//        this.splitController = temp.splitController;
//        this.usedAttributes = temp.usedAttributes;
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
    myC45 finalTree = new myC45();
    //finalTree = this;
    //finalTree = pruneTree(finalTree);
    //copy(finalTree);
  }

  /**
   * Method for building a C45 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, Default default_data, ArrayList<Attribute> used) throws Exception {
    // Initialize variables
    dataset = data;
    usedAttributes = new ArrayList<>();
    if (used != null){
        usedAttributes = used;
        for (int i = 0; i < usedAttributes.size(); i++) {
            dataset.deleteAttributeAt(usedAttributes.get(i).index());
        }
    }
    numericValues = new double[dataset.numAttributes()];
    //System.out.println(data);
    
    // Check if no instances have reached this node. Missing Value
    if (dataset.numInstances() == 0) {
      main_Attribute = null;
      m_ClassValue = default_data.classValue();
      m_ClassAttribute = default_data.classAttribute();
      class_Distribution = new double[dataset.numClasses()];
      return;
    }
    
    //Make leaf if all attributes has been used, use most common class
    if (usedAttributes.size() == dataset.numAttributes()) {
      class_Distribution = new double[dataset.numClasses()];
      Enumeration classEnum = dataset.enumerateAttributes();
      while (classEnum.hasMoreElements()) {
         Instance inst = (Instance) classEnum.nextElement(); 
         class_Distribution[(int) inst.classIndex()]++;
      }  
      double chosenClass = Utils.maxIndex(class_Distribution);
      main_Attribute = null;
      m_ClassValue = chosenClass;
      m_ClassAttribute = dataset.classAttribute();
      return;  
    }
        
    // Compute attribute with maximum information gain.
    // If attribute is continuous valued, define discrete values for attribute
    double[] infoGains = new double[dataset.numAttributes()];
    double[] gainRatios = new double[dataset.numAttributes()];
    Enumeration attEnum = dataset.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      if (att.isNumeric()) {
        infoGains[att.index()] = calculateNumeric(dataset, att);
        gainRatios[att.index()] = infoGains[att.index()] / 
            computeSplitInfoNumeric(dataset, att, numericValues[att.index()]);
      } else {
        infoGains[att.index()] = computeInfoGain(dataset, att);
        gainRatios[att.index()] = infoGains[att.index()] / computeSplitInfo(dataset,att);
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
      class_Distribution = getClassDistribution(this);
      m_ClassValue = Utils.maxIndex(class_Distribution);
      m_ClassAttribute = dataset.classAttribute();
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
    
  /**
   * Calculate class distribution of node
   *
   * @param node the node which class distribution is to be calculated
   * @return the class distribution of the node
   */
    private double[] getClassDistribution(myC45 node) {
        double [] dist = new double[node.dataset.numClasses()];
        Enumeration instEnum = node.dataset.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            dist[(int) inst.classValue()]++;
        }
        Utils.normalize(dist);
        return dist;
    }
  
    /**
   * Get child class values of node
   *
   * @param subTree the node of which it's child is to be retrieved
   * @return the array of values of all child classes
   */
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
    
    /**
   * Get most common class of node determined by it's child
   *
   * @param subTree the node which it's most common class is to be determined
   * @return the most common class' value
   */
    private Double getMajorityClass(myC45 subTree) {
        ArrayList<Double> childValues = getChildClasses(subTree);
        HashMap<Double,Integer> hm = new HashMap<>();
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
    
    /**
   * Return the result of pruning input subtree according to child index
   *
   * @param inNode the node which it's child is going to be pruned
   * @param index the index of child that is going to be pruned
   * @return the pruned child of input node according to index
   */
    private myC45 alterChild(myC45 inNode, int index) {
        myC45 child = inNode.child_Nodes[index];
        child.main_Attribute = null;
        child.child_Nodes = null;
        child.m_ClassValue = getMajorityClass(child);
        child.m_ClassAttribute = child.dataset.classAttribute();
        child.class_Distribution = getClassDistribution(child);
        return child;
    }

     /**
   * Return pruned tree or iterate through tree to prune subtrees recursively
   *
   * @param node the tree node which is going to be pruned
   * @return pruned tree
   */
    private myC45 pruneTree(myC45 node) throws Exception{
        if (node.main_Attribute == null) {} 
        else {
            for (int i = 0; i < node.child_Nodes.length; i++) {
                Evaluation eval = new Evaluation(node.dataset);
                eval.evaluateModel(node, node.dataset);
                double initError = eval.errorRate();
                myC45 altered = alterChild(node,i);
                node.child_Nodes[i] = altered;
                eval.evaluateModel(node, node.dataset);
                double endError = eval.errorRate();
                if (Utils.gr(initError, endError)) {
                    node.child_Nodes[i] = altered;
                } else {
                    node.child_Nodes[i] = pruneTree(node.child_Nodes[i]);
                }
            }
        }
        return node;
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
