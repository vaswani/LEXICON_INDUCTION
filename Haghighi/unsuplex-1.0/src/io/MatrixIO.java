package io;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class MatrixIO {
	
    static final String INPUT_ENC = "UTF-8";
    static final String OUTPUT_ENC = "UTF-8";
	
    public static void writeMatrix(double[][] mat, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), OUTPUT_ENC));
			for (double[] row : mat) {
				for (double val : row) {
					out.write(val + "\t");
				}
				out.newLine();
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

}
