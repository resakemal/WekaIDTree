/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3;

import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Resa Kemal Saharso
 */
public class SplitData {
    
    private Instances[] split;
    private int mostCommon;
    
    /**
   * Returns split data.
   *
   * @return result of split
   */
    public Instances[] getSplit() {
        return split;
    }
    
    /**
   * Returns most common value index.
   *
   * @return result of split
   */
    public int getMostCommon() {
        return mostCommon;
    }
   
   /**
   * Splits a dataset according to the values of a nominal attribute.
   *
   * @param data the data which is to be split
   * @param att the attribute to be used for splitting
   */
    public void splitData(Instances data, Attribute att) {

        split = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
          split[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          if (!inst.isMissing(att)) {
            split[(int) inst.value(att)].add(inst);
          }
        }
          for (Instances splitData1 : split) {
              splitData1.compactify();
          }
    }
    
    /**
   * Splits a dataset according to a numeric value into 2 boolean instances
   *
   * @param data the data which is to be split
   * @param att the attribute to be used for splitting
   * @param value the value of numeric attribute for split
   */
    public void splitDataNumeric(Instances data, Attribute att, double value) {
        
        split = new Instances[2];
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
    }
    
     /**
   * Add missing values to most common attribute value
   *
   * @param data the data which is missing value is from
   * @param att the attribute which most common value is determined
   */
    public void handleMissingValues(Instances data, Attribute att) {
        double [] valueDist;
        valueDist = new double[att.numValues()];
        Instances missingInst = new Instances(data, 0);
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if(inst.isMissing(att)) {
                missingInst.add(inst);
            } else {
                valueDist[(int) inst.value(att)]++;
            }
        }
        mostCommon = Utils.maxIndex(valueDist);
        Enumeration missEnum = missingInst.enumerateInstances();
        while (missEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            split[mostCommon].add(inst);
        }
    }
    
     /**
   * Add missing values to most common attribute value
   *
   * @param data the data which is missing value is from
   * @param att the attribute which most common value is determined
   */
    public void handleMissingValuesNumeric(Instances data, Attribute att, double value) {
        double [] valueDist;
        valueDist = new double[2];
        Instances missingInst = new Instances(data, 0);
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if(inst.isMissing(att)) {
                missingInst.add(inst);
            } else {
                if (inst.value(att) <= value) {
                    valueDist[0]++;
                } else {
                    valueDist[1]++;
                }
            }
        }
        mostCommon = Utils.maxIndex(valueDist);
        Enumeration missEnum = missingInst.enumerateInstances();
        while (missEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            split[mostCommon].add(inst);
        }
    }
}
