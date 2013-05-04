package inducer;

import java.util.ArrayList;
import java.util.List;

import canco.BipartiteMatcher;

import fig.basic.*;
import fig.exec.*;
import static fig.basic.LogInfo.*;

import kernelcca.*;

public class SCAInducerTest {
  public static void evalKernel() throws Exception {
    String line;
    while((line = LogInfo.stdin.readLine()) != null) {
      String[] tokens = line.split(" ");
      logs(new StringKernel().dot(tokens[0], tokens[1]));
    }
  }

	public static void main(String[] args) throws Exception {
    KernelCCA.Options opts = new KernelCCA.Options();
		Execution.init(args, opts);

		//String[] names = {"aria", "percy","dan","john","slav","adam","alex"};
		//String[] names2 = {"daria", "perci","man","johl","slave","madam","alx"};
		//String[] names = {"Commission"};
		//String[] names2 = {"Comisiun"};
    //evalKernel();
    
		//String[] names = {"Commission", "Consejo"};
		//String[] names2 = {"Comisiun", "Consejo"};
		String[] names = {"Commission", "Parliament"};
		String[] names2 = {"Comisiun", "Parlamento"};
					
		Kernel<String> strKernel = new StringKernel();
		KernelCCA<String> kernelCCA = new KernelCCA<String>(opts);
		List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		for (int i=0; i < names.length; ++i) {			
			pairs.add(Pair.newPair(names[i], names2[i]));
		}		
		kernelCCA.setData(pairs, strKernel, strKernel, strKernel);
		Kernel<String> solKernel = kernelCCA.solve();
		VectorRepresenter<String> domRepn = kernelCCA.getXRepresentation();
		VectorRepresenter<String> codomRepn = kernelCCA.getYRepresentation();
		BipartiteMatcher matcher = new BipartiteMatcher(names.length);
		for (int i=0; i < names.length; ++i) {
      logs("X: " + names[i] + " is " + Fmt.D(domRepn.getRepn(names[i])));
      logs("Y: " + names2[i] + " is " + Fmt.D(codomRepn.getRepn(names2[i])));
    }

		for (int i=0; i < names.length; ++i) {
			for (int j=0; j < names.length; ++j) {
        logs(names[i] + " " + names2[j] + " => " + solKernel.dot(names[i],names2[j]));
				matcher.setWeight(i,j,solKernel.dot(names[i],names2[j]));
			}
		}
		int[] matching = matcher.getMatching();
		for (int i=0; i < matching.length; ++i) {
			int j = matching[i];
			logs("%s => %s\n",names[i],names2[j]);
		}
		Execution.finish();
	}
}
