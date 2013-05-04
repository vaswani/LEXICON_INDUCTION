package inducer;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.mapper.MapWorker;
import edu.berkeley.nlp.mapper.MapWorkerFactory;
import edu.berkeley.nlp.mapper.Mapper;
import edu.berkeley.nlp.util.Factory;

public class DistributedMatchingMatrixFiller {
	
	public static interface EntryFiller {
		public double getEntry(int i, int j); 
	}
		
	public static void fillMatrix(final double[][] matrix, final Factory<EntryFiller> fillerFactory) {
		MapWorkerFactory<Integer> mapWorkerFactory = new MapWorkerFactory<Integer>() {
			
			class Worker extends MapWorker<Integer> {
				EntryFiller entryFiller;
				Worker(EntryFiller entryFiller) {
					this.entryFiller = entryFiller;
				}
				@Override
				public void map(Integer item) {
					int i = item;
					double[] row = matrix[i];
					for (int j=0; j < row.length; ++j) {
						row[j] = entryFiller.getEntry(i, j);  
					}
				}				
			}

			public MapWorker<Integer> newMapWorker() {
				Worker worker = new Worker(fillerFactory.newInstance());
				return worker;
			}
			
		};
		Mapper mapper = new Mapper<Integer>(mapWorkerFactory);
		int n = matrix.length;
		List<Integer> rowItems = new ArrayList<Integer>();
		for (int i=0; i < n; ++i) { rowItems.add(i); }
		mapper.doMapping(rowItems);
	}
	
}
