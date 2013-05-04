package io;

import fig.basic.Pair;
import io.POSTagPrefixes.POSTag;

import java.util.List;
import java.util.Set;

public interface BitextProcessor {
	public void setBitext(Bitext bitext) ;
	public void setPOSTags(Set<POSTag> posTags) ;
	public Pair<List<List<String>>, List<List<String>>> getReducedCorproa();
}
