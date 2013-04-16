package visualization;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Shape;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.lowagie.text.Document;
import com.lowagie.text.pdf.PdfContentByte;
import com.lowagie.text.pdf.PdfWriter;


import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.Pair;

public class CanonicalSpacePDFPrinter {
	static final Color DOM_COLOR = Color.red;
	static final Color CODOM_COLOR = Color.blue;
	static final Color BACKGROUND_COLOR = Color.white;
	static final Color EDGE_COLOR = Color.gray;
	static final Color LABEL_COLOR = Color.black;
	static final double LINE_WEIGHT = 1.0/8.2;
	static final double BORDER = 1.0/20.0;
	static final double LABEL_BORDER = -1.0/2.7;
	static final double MIN_RADIUS = 1.0/150.0;
//	static final double MIN_RADIUS = 1.0/20.0;
	static final double MAX_RADIUS = 1.0/10.0;
	static final int DOC_DIM = 500;

	private Representer2D repr;

	public CanonicalSpacePDFPrinter(Representer2D repr) {
		this.repr = repr;
	}
	
	// writes first numPairs translations from non-seed source words
	public void writePDF(Pair<double[][],double[][]> reps, Indexer<String> domWords, Indexer<String> codomWords, Map<String,String> dict, CounterMap<String,String> seed, int numPairs, String fileName) {
		List<Pair<Double,Double>> domReps = repr.get2DRepresentations(reps.getFirst());
		List<Pair<Double,Double>> codomReps = repr.get2DRepresentations(reps.getSecond());
		List<Pair<Double,Double>> newDomRep = new ArrayList<Pair<Double,Double>>();
		List<Pair<Double,Double>> newCodomRep = new ArrayList<Pair<Double,Double>>();
		Indexer<String> newDomWords = new Indexer<String>();
		Indexer<String> newCodomWords = new Indexer<String>();
		int soFar = 0;
		for (int i = 0; i < domWords.size() && soFar < numPairs; i++) {
			String domWord = domWords.getObject(i);
			if (seed.keySet().contains(domWord)) continue;
			if (!codomWords.contains(dict.get(domWord))) continue;
			
			newDomWords.add(domWord);
			newDomRep.add(domReps.get(i));
			
			String codomWord = dict.get(domWord);
			newCodomWords.add(codomWord);
			newCodomRep.add(codomReps.get(codomWords.getIndex(codomWord)));
			
			soFar++;
		}
		writePDF(newDomRep, newCodomRep, newDomWords, newCodomWords, dict, fileName);
	}
	
	public void writePDF(Pair<double[][],double[][]> reps, Indexer<String> domWords, Indexer<String> codomWords, Map<String,String> dict, String fileName) {
		List<Pair<Double,Double>> domReps = repr.get2DRepresentations(reps.getFirst());
		List<Pair<Double,Double>> codomReps = repr.get2DRepresentations(reps.getSecond());
		writePDF(domReps, codomReps, domWords, codomWords, dict, fileName);
	}

	private void writePDF(List<Pair<Double,Double>> domReps, List<Pair<Double,Double>> codomReps, Indexer<String> domWords, Indexer<String> codomWords, Map<String,String> dict, String fileName) {
		
		com.lowagie.text.Rectangle docDims = new com.lowagie.text.Rectangle(DOC_DIM, DOC_DIM);
		Document document = new Document(docDims);

		try {
			// get pdf writer
			PdfWriter writer = PdfWriter.getInstance(document, new FileOutputStream(fileName));
			document.open();
			PdfContentByte cb = writer.getDirectContent();
			java.awt.Graphics2D g2 = cb.createGraphicsShapes(DOC_DIM, DOC_DIM);

			// get data scale
			List<Pair<Double,Double>> allReps = new ArrayList<Pair<Double,Double>>(domReps);
			allReps.addAll(codomReps);
			Pair<Pair<Double,Double>,Pair<Double,Double>> bounds = getRepBounds(allReps);
			Pair<Double,Double> dataXBounds = bounds.getFirst();
			Pair<Double,Double> dataYBounds = bounds.getSecond();
			double dataXCenter = (dataXBounds.getFirst()+dataXBounds.getSecond())/2.0;
			double dataYCenter = (dataYBounds.getFirst()+dataYBounds.getSecond())/2.0;
			double dataMaxSpan = Math.max(dataXBounds.getSecond()-dataXBounds.getFirst(), dataYBounds.getSecond()-dataYBounds.getFirst());

			// get radius size
			double minDist = getRepMinDist(allReps);
			double radius = Math.min(Math.max(minDist / 2.0, MIN_RADIUS*dataMaxSpan), MAX_RADIUS*dataMaxSpan);

			// calculate screen dimensions in data space
			double span = dataMaxSpan+2*radius+2*BORDER*dataMaxSpan;
			Pair<Double,Double> xBounds = Pair.newPair(dataXCenter-(span/2.0), dataXCenter+(span/2.0));
			Pair<Double,Double> yBounds = Pair.newPair(dataYCenter-(span/2.0), dataYCenter+(span/2.0));

			// set translation from data space to screen space for graph (need to flip y-axis)
			AffineTransform graphAff = new AffineTransform();
			graphAff.translate(0.0, ((double)DOC_DIM));
			graphAff.scale(((double)DOC_DIM)/span,-((double)DOC_DIM)/span);
			graphAff.translate(-xBounds.getFirst(), -yBounds.getFirst());


			// set translation from data space to screen space for text (flip y-axis second time: right-side up)
			AffineTransform textAff = new AffineTransform();
			textAff.translate(0.0, ((double)DOC_DIM));
			textAff.scale(((double)DOC_DIM)/span,-((double)DOC_DIM)/span);
//			textAff.scale(1,-1);
			textAff.translate(-xBounds.getFirst(), -yBounds.getFirst());

			g2.setTransform(graphAff);

			// make graphics objects
			List<Line2D> edges = getEdges(domReps, codomReps, domWords, codomWords, dict);
			List<Ellipse2D> domNodes = getNodes(domReps, radius);
			List<Ellipse2D> codomNodes = getNodes(codomReps, radius);

			g2.setStroke(new BasicStroke((float) ((LINE_WEIGHT)*radius), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

			// draw edges
			g2.setPaint(EDGE_COLOR);
			for (Line2D e : edges) {
				g2.draw(e);
			}

			// draw domain nodes
			g2.setPaint(DOM_COLOR);
			for (Ellipse2D n : domNodes) {
				g2.draw(n);
			}

			// draw codomain nodes
			g2.setPaint(CODOM_COLOR);
			for (Ellipse2D n : codomNodes) {
				g2.draw(n);
			}

			// draw data labels
			g2.setPaint(LABEL_COLOR);
			g2.setTransform(textAff);
			Font font = new Font("Sans-serif", Font.PLAIN, 10);
			FontRenderContext frc = g2.getFontRenderContext();

			// get font size transfrom
			Set<String> allWords = new HashSet<String>();
			allWords.addAll(domWords);
			allWords.addAll(codomWords);
			double maxLabelWidth = getMaxLabelWidth(allWords, font, frc);
			AffineTransform fontScaleAff = new AffineTransform();
			double labelBorder = LABEL_BORDER*(2*radius);
			double desiredMaxLabelWidth = 2*(radius)-2*labelBorder;
			fontScaleAff.scale(desiredMaxLabelWidth/maxLabelWidth,-desiredMaxLabelWidth/maxLabelWidth);

			for (String s : domWords) {
				// get centering transform and draw
				int nodeNum = domWords.getIndex(s);
				GlyphVector gv = font.createGlyphVector(frc, s);
				Shape shape = gv.getOutline();
				shape = fontScaleAff.createTransformedShape(shape);
				Rectangle2D shapeBds = shape.getBounds2D();
				AffineTransform centeringAff = new AffineTransform();
				centeringAff.translate(domReps.get(nodeNum).getFirst()-(shapeBds.getWidth()/2.0), domReps.get(nodeNum).getSecond()-(shapeBds.getHeight()/2.0));
				shape = centeringAff.createTransformedShape(shape);
				g2.fill(shape);
			}

			for (String s : codomWords) {
				// get centering transform and draw
				int nodeNum = codomWords.getIndex(s);
				GlyphVector gv = font.createGlyphVector(frc, s);
				Shape shape = gv.getOutline();
				shape = fontScaleAff.createTransformedShape(shape);
				Rectangle2D shapeBds = shape.getBounds2D();
				AffineTransform centeringAff = new AffineTransform();
				centeringAff.translate(codomReps.get(nodeNum).getFirst()-(shapeBds.getWidth()/2.0), codomReps.get(nodeNum).getSecond()-(shapeBds.getHeight()/2.0));
				shape = centeringAff.createTransformedShape(shape);
				g2.fill(shape);
			}

			g2.dispose();
			document.close();
		}

		catch (Exception de) {
			de.printStackTrace();
		}
	}

	private static double getRepMinDist(List<Pair<Double,Double>> allReps) {
		double minDist = Double.POSITIVE_INFINITY;
		for (Pair<Double,Double> rep1 : allReps) {
			for (Pair<Double,Double> rep2 : allReps) {
				if (rep1 != rep2) {
					double dist = Math.sqrt(Math.pow(rep1.getFirst() - rep2.getFirst(), 2) + Math.pow(rep1.getSecond() - rep2.getSecond(), 2));
					minDist = Math.min(minDist, dist);
				}
			}
		}
		return minDist;
	}


	private static Pair<Pair<Double,Double>,Pair<Double,Double>> getRepBounds(List<Pair<Double,Double>> allReps) {
		double minX = Double.POSITIVE_INFINITY;
		double maxX = Double.NEGATIVE_INFINITY;
		double minY = Double.POSITIVE_INFINITY;
		double maxY = Double.NEGATIVE_INFINITY;
		for (Pair<Double,Double> rep1 : allReps) {
			minX = Math.min(minX, rep1.getFirst());
			maxX = Math.max(maxX, rep1.getFirst());
			minY = Math.min(minY, rep1.getSecond());
			maxY = Math.max(maxY, rep1.getSecond());
		}
		return Pair.newPair(Pair.newPair(minX, maxX), Pair.newPair(minY, maxY));
	}

	private List<Ellipse2D> getNodes(List<Pair<Double,Double>> reps, double radius) {
		List<Ellipse2D> ellps = new ArrayList<Ellipse2D>();
		for (Pair<Double,Double> rep : reps) {
			ellps.add(new Ellipse2D.Double(rep.getFirst()-radius, rep.getSecond()-radius, 2.0*radius, 2.0*radius));
		}
		return ellps;
	}

	private List<Line2D> getEdges(List<Pair<Double,Double>> domReps, List<Pair<Double,Double>> codomReps, Indexer<String> domWords, Indexer<String> codomWords, Map<String,String> dict) {
		List<Line2D> lines = new ArrayList<Line2D>();
		for (String domWord : domWords) {
			if (!dict.containsKey(domWord)) continue;
			String codomWord = dict.get(domWord);
			if (!codomWords.contains(codomWord)) continue;
			Pair<Double,Double> pt1 = domReps.get(domWords.getIndex(domWord));
			Pair<Double,Double> pt2 = codomReps.get(codomWords.getIndex(codomWord));
			lines.add(new Line2D.Double(pt1.getFirst(), pt1.getSecond(), pt2.getFirst(), pt2.getSecond()));
		}
		return lines;
	}

	private static double getMaxLabelWidth(Set<String> allWords, Font font, FontRenderContext frc) {
		double maxLabelWidth = Double.NEGATIVE_INFINITY;
		for (String s : allWords) {
			GlyphVector gv = font.createGlyphVector(frc, s);
			Shape shape = gv.getOutline();
			Rectangle2D shapeBds = shape.getBounds2D();
			maxLabelWidth = Math.max(maxLabelWidth, shapeBds.getWidth());
		}
		return maxLabelWidth;
	}

	public static void main(String[] args) {

		double[][] domReps = new double[5][];
		domReps[0] = new double[] {0.0, 0.0, -10.3, 70, 0};
		domReps[1] = new double[] {10.0, 0.0, -10.3, 70, 0};
		domReps[2] = new double[] {0.0, -10.0, -10.3, 70, 0};
		domReps[3] = new double[] {-100.0, 0.0, -10.3, 70, 0};
		domReps[4] = new double[] {0.0, 100.0, -10.3, 70, 0};

		double[][] codomReps = new double[5][];
		codomReps[0] = new double[] {0.0, 0.0, -10.3, 70, 0};
		codomReps[1] = new double[] {0.0, 0.0, -10.3, 70, 0};
		codomReps[2] = new double[] {0.0, 0.0, -10.3, 70, 0};
		codomReps[3] = new double[] {150.0, 0.0, -10.3, 70, 0};
		codomReps[4] = new double[] {0.0, -150.0, -10.3, 70, 0};

		Indexer<String> domWords = new Indexer<String>();
		domWords.add("cat");
		domWords.add("city");
		domWords.add("library");
		domWords.add("Germany");
		domWords.add("dog");

		Indexer<String> codomWords = new Indexer<String>();
		codomWords.add("gato");
		codomWords.add("ciudad");
		codomWords.add("biblioteca");
		codomWords.add("Alemenia");
		codomWords.add("Perro");

		Map<String,String> dict = new HashMap<String,String>();
		dict.put("cat", "gato");
		dict.put("city", "ciudad");
		dict.put("library", "biblioteca");
		dict.put("Germany", "Alemenia");
		dict.put("dog", "Perro");

		CanonicalSpacePDFPrinter printer = new CanonicalSpacePDFPrinter(new First2Representer2D());
		printer.writePDF(Pair.newPair(domReps, codomReps), domWords, codomWords, dict, "testWritePDF.pdf");

	}


}
