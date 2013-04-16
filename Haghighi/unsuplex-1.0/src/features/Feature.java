package features;

public class Feature {
	
	private String str ;
	private int index ;
	
	public Feature(String str, int index) {
		super();
		this.str = str;
		this.index = index;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((str == null) ? 0 : str.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		final Feature other = (Feature) obj;
		if (str == null) {
			if (other.str != null)
				return false;
		} else if (!str.equals(other.str))
			return false;
		return true;
	}
	
	public String toString() {
		return str;
	}
	
	public int getIndex() {
		return index;
	}

}
