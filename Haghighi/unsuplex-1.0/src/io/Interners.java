package io;

import fig.basic.Interner;


public class Interners {
	//static final Interner<Pair<String,String>> pairInterner = new Interner<Pair<String,String>>();
	public static final Interner<String> stringInterner = new Interner<String>();
}
