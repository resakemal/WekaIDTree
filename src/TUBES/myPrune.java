package TUBES;

/**
 *
 * @author JJ
 */
import java.util.ArrayList;
import java.util.HashMap;

public class myPrune {

    private ArrayList<Double> getChildClasses(myC45 subTree) {
        // If node is not leaf, get child's class value
        ArrayList<Double> childValues = new ArrayList<>();
        if (subTree.mainAttribute() != null) {
            for (int i = 0; i < subTree.child_Nodes().length; i++) {
                childValues.addAll(getChildClasses(subTree.child_Nodes()[i]));
            }
        } else {
            childValues.add(subTree.classValue());
        }
        return childValues;
    }
    
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
}
