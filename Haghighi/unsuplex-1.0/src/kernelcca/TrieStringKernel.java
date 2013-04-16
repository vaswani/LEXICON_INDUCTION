package kernelcca;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrieStringKernel implements Kernel<String> {
	
	int k = 5;
	double alpha = 0.9;

	public double dot(String x, String y) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	private TrieNode makeInitialNode(String s) {
		TrieNode node = new TrieNode();
		for (int i=0; i <= s.length()-k; ++i) {
			String sub = s.substring(i,i+k);
			
		}
		return node;
	}
	
	Map<String, TrieNode> trieNodeMap = new HashMap<String, TrieNode>(); 
	
	class TrieNode {
		class TrieElem {
			String substr ;
			int depth ;
			private TrieElem(String substr, int depth) {
				super();
				this.substr = substr;
				this.depth = depth;
			}			
		}
		List<TrieElem> elems = new ArrayList<TrieElem>();
	}

}
