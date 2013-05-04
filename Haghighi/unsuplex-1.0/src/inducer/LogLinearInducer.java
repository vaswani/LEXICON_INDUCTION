package inducer;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import static fig.basic.LogInfo.*;

import io.POSTagPrefixes.POSTag;
import kernelcca.*;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Triple;

public class LogLinearInducer implements DictionaryInducer {
  public enum ModelType { forw, back, forw2, back2 };
  public enum InitType { truePred, randomPred, seedPred, editDist };
  public enum FeatureType { editDist, editDistInd, wcluster, freq, cheat, cholesky, cca, tmp };

  public static class Options {
    @Option public int numIters = 10;
    @Option(gloss="Number of reduction steps when optimizing") public int numReductionSteps = 10;
    @Option public double regularization = 0;
    @Option(gloss="Type of initialization") public InitType initType = InitType.randomPred;
    @Option public double initStepSize = 1;
    @Option(gloss="Temperature (when 0, get MAP)") public double temperature = 1;
    @Option(gloss="Whether to use seed mapping to learn") public boolean useSeed = false;
    @Option(gloss="Whether to permit null-aligned words") public boolean allowNull = false;
    @Option public ArrayList<ModelType> models = ListUtils.newList(ModelType.forw, ModelType.back);
    @Option public ArrayList<FeatureType> features = ListUtils.newList(FeatureType.editDist);
    @Option public Random initRandom = new Random(1);
    @Option public boolean tmpExample;
    @Option public double convergenceThreshold = 1e-4;
    @Option public int maxCholeskyBases = 10;
    @Option(gloss="When learning, always get seed predictions right") public boolean clampToSeed = true;
  }

  /*abstract class Feature {
    String name;
    public Feature(String name) { this.name = name; }
    double eval(int i, int j) { return i < L1.N && j < L2.N ? eval(L1.wstr(i), L2.wstr(j)) : 0; }
    double eval(String w1, String w2) { return 0; } // OVERRIDE
    public toString() { return name; }
  }*/

  static class Language {
    int N; // Number of words
    Indexer<String> words;
    HashMap<String,Double>[] features; // Monolingual features
    int[] freq;
    int sumFreq;

    double[][] choleskyRepn;
    int B;
    //double[][] ccaRepn;

    String wstr(int i) { return i == N ? "(null)" : words.getObject(i); }

    List<List<String>> sentences; // Raw sentences
    Map<String, Set<POSTag>> posMap; // POS of each word

    public Language(Indexer<String> words, List<List<String>> sentences, Map<String,Set<POSTag>> posMap) {
      this.words = words;
      this.sentences = sentences;
      this.posMap = posMap;
      this.N = words.size();
      this.freq = new int[N];
      for(List<String> sentence: sentences) {
        for(String word : sentence) {
          int i = words.indexOf(word);
          if(i != -1) freq[i]++;
        }
      }
      this.sumFreq = ListUtils.sum(freq);
    }

    public void createCholeskyRepn(Kernel<String> kernel, int maxB) {
      this.B = Math.min(maxB, N);
      this.choleskyRepn = new double[N][];
      VectorRepresenter<String> repn = new IncompleteCholeskyDecomposition<String>(kernel, (String[])words.getObjects().toArray(new String[0]), 0.001);
      for(int i = 0; i < N; i++)
        choleskyRepn[i] = repn.getRepn(i);
    }
  }

  // Input
  Options opts;
  DictionaryInducerTester.Options dopts = DictionaryInducerTester.opts;
  Language L1, L2;
  int[] seedMap1, seedMap2; // i -> j that is matched or -1; j -> i that is matched or -1
  int numSeeds;
  Map<String, Set<String>> trueTranslations; // Ground truth
  boolean useForwModel, useBackModel;

  HashMap<String,Double>[][] biFeatures;

  // M-step quantities
  HashMap<String,Double> params; // theta
  double[][] forwProbs, backProbs; // deterministic function of theta
  int paramsVersion = 0, paramProbsVersion = -1; // Make sure params and probs are in sync
  double stepSize;
  double logLikelihood;

  // E-step quantities
  double[][] mu1, mu2; // i, j -> probability that i and j are aligned under the models
  double entropy;

  double mu(int i, int j) {
    return (useForwModel && j < L2.N ? mu1[i][j] : 0) + (useBackModel && i < L1.N ? mu2[i][j] : 0);
  }

  // Diagnostic
  void checkQuadraticEffectiveness() {
  }

  /*void investigateFeature(Feature f) {
    FullStatFig transFig = new FullStatFig();
    FullStatFig notTransFig = new FullStatFig();
    for(int i = 0; i <= L1.N; i++) {
      for(int j = 0; j <= L2.N; j++) {
        if(isTrueTranslation(i, j))
          transFig.add(f.eval(i, j));
        else
          notTransFig.add(f.eval(i, j));
      }
    }
    logs("%s: trans [%s], notTrans [%s]", f, transFig, notTransFig);
  }*/

  boolean useCCA() { return opts.features.contains(FeatureType.cca); }
  boolean useCholesky() { return opts.features.contains(FeatureType.cholesky); }

  double freqDistance(int i, int j) {
    double freq1 = 1.0 * L1.freq[i] / L1.sumFreq;
    double freq2 = 1.0 * L2.freq[i] / L2.sumFreq;
    return Math.abs(freq1-freq2) / (0.5 * (freq1+freq2));
  }

  void investigateFeatures() {
    track("investigateFeatures");
    HashMap<String, FullStatFig> transFigs = new HashMap();
    HashMap<String, FullStatFig> notTransFigs = new HashMap();

    for(int i = 0; i <= L1.N; i++) {
      for(int j = 0; j <= L2.N; j++) {
        boolean isTrans = isTrueTranslation(i, j);
        for(Map.Entry<String,Double> pair : biFeatures[i][j].entrySet()) {
          if(!transFigs.containsKey(pair.getKey())) transFigs.put(pair.getKey(), new FullStatFig());
          if(!notTransFigs.containsKey(pair.getKey())) notTransFigs.put(pair.getKey(), new FullStatFig());
          if(isTrans) transFigs.get(pair.getKey()).add(pair.getValue());
          else notTransFigs.get(pair.getKey()).add(pair.getValue());
        }
      }
    }
    ArrayList<String> features = new ArrayList();
    for(String f : transFigs.keySet())
      features.add(f); for(String f : notTransFigs.keySet())
      if(!transFigs.containsKey(f)) features.add(f);

    double[] goodness = new double[features.size()];
    for(int i = 0; i < features.size(); i++)
      goodness[i] = Math.abs(transFigs.get(features.get(i)).mean() - notTransFigs.get(features.get(i)).mean());

    for(int i : ListUtils.sortedIndices(goodness, true))
      logs("%s (%s): trans [%s], notTrans [%s]", features.get(i), Fmt.D(goodness[i]),
          transFigs.get(features.get(i)), notTransFigs.get(features.get(i)));
    end_track();
  }

  public LogLinearInducer(Options opts, Indexer<String> words1, Indexer<String> words2,
      CounterMap<String,String> seedMapping, NewBitext bitext) {
    this.opts = opts;
    // Add seed mapping to words
    words1 = new Indexer(words1);
    words2 = new Indexer(words2);
    for(String w1 : seedMapping.keySet()) {
      words1.getIndex(w1);
      for(String w2 : seedMapping.getCounter(w1).keySet())
        words2.getIndex(w2);
    }

    // TMP
    if(opts.tmpExample) {
      words1 = new Indexer();
      words1.getIndex("Parliament");
      words1.getIndex("Comission");
      words2 = new Indexer();
      words2.getIndex("Parlement");
      words2.getIndex("Comission");
    }

    this.L1 = new Language(words1, bitext.domainCorpus, bitext.domainPOSMap);
    this.L2 = new Language(words2, bitext.codomainCorpus, bitext.codomainPOSMap);
    this.trueTranslations = bitext.bilingualLex.map;
    this.useForwModel = opts.models.contains(ModelType.forw);
    this.useBackModel = opts.models.contains(ModelType.back);

    // Setup Cholesky representation
    if(useCholesky() || useCCA()) {
      track("Setup cholesky representation based on kernel");
      Triple<Kernel<String>, Kernel<String>, Kernel<String>> bestKernels =
        BestKernel.getBestKernel(new Pair(words1, words2), bitext, dopts);
      L1.createCholeskyRepn(bestKernels.getFirst(), opts.maxCholeskyBases);
      L2.createCholeskyRepn(bestKernels.getSecond(), opts.maxCholeskyBases);
      end_track();
    }

    // Allow efficient testing of seeds
    this.numSeeds = 0;
    this.seedMap1 = ListUtils.newInt(L1.N, -1);
    this.seedMap2 = ListUtils.newInt(L2.N, -1);
    for(int i = 0; i < L1.N; i++) {
      for(int j = 0; j < L2.N; j++) {
        if(seedMapping.getCount(L1.wstr(i), L2.wstr(j)) > 0) {
          assert seedMap1[i] == -1; seedMap1[i] = j;
          assert seedMap2[j] == -1; seedMap2[j] = i;
          numSeeds++;
        }
      }
    }
    logs("%d seed pairs", numSeeds);

    checkQuadraticEffectiveness();
  }

  boolean isTrueTranslation(int i, int j) {
    if(i == L1.N || j == L2.N) return false;
    String w1 = L1.wstr(i), w2 = L2.wstr(j);
    if(!trueTranslations.containsKey(w1)) return false;
    return trueTranslations.get(w1).contains(w2);
  }
  boolean isSeed(int i, int j) { return i < L1.N && seedMap1[i] == j; }
  boolean involvedInSeed(int i, int j) { return (i < L1.N && seedMap1[i] != -1) || (j < L2.N && seedMap2[j] != -1); }

  public void extractFeatures() {
    track("extractFeatures");
    this.biFeatures = new HashMap[L1.N+1][L2.N+1];
    //ArrayList<Feature> features = new ArrayList();
    //features.add(new Feature("editDistance") { double eval(String w1, String w2) { return EditDistanceInducer.editDistance(w1, w2); } });

    for(int i = 0; i <= L1.N; i++) {
      for(int j = 0; j <= L2.N; j++) {
        HashMap<String,Double> features = biFeatures[i][j] = new HashMap();
        if(i == L1.N && j == L2.N)
          ;
        else if(i == L1.N || j == L2.N)
          ; //features.put("nullAligned", 1.0);
        else {
          String w1 = L1.wstr(i);
          String w2 = L2.wstr(j);

          for(FeatureType ft : opts.features) {
            switch(ft) {
              case editDist:
                features.put("editDist", EditDistanceInducer.editDistance(w1, w2)/Math.max(w1.length(), w2.length()));
                break;
              case editDistInd:
                // Hurts
                for(int d = 0; d < 4; d++)
                  addIndicator(features, "editDist="+d, EditDistanceInducer.editDistance(w1, w2) == d);
                break;
              case freq:
                features.put("freqDist", freqDistance(i, j)); // Does nothing
                break;
              case cheat:
                addIndicator(features, "isCorrect", isTrueTranslation(i, j)); // Cheating!
                break;
              case cholesky:
                for(int b = 0; b < L1.B; b++)
                  for(int c = 0; c < L2.B; c++)
                    features.put("cholesky-"+b+"-"+c, L1.choleskyRepn[i][b]*L2.choleskyRepn[j][c]);
                break;
              case tmp:
                addIndicator(features, "Parliament", "Parlement", w1, w2);
                addIndicator(features, "Mr", "Monsieur", w1, w2);
                break;
              default:
                throw Exceptions.unknownCase;
            }
          }
        }
      }
    }
    end_track();

    investigateFeatures();
  }
  void addIndicator(HashMap<String,Double> features, String q1, String q2, String w1, String w2) {
    if(w1.equals(q1) && w2.equals(q2)) features.put(q1+"<->"+q2, 1.0);
  }
  void addIndicator(HashMap<String,Double> features, String name, boolean b) {
    if(b) features.put(name, 1.0);
  }

  public double getScore(int i, int j) {
    if(opts.clampToSeed && involvedInSeed(i, j))
      return isSeed(i, j) ? 0 : Double.NEGATIVE_INFINITY;

    double score = 0;
    for(Map.Entry<String,Double> pair : biFeatures[i][j].entrySet())
      score += MapUtils.get(params, pair.getKey(), 0.0) * pair.getValue();
    return score;
  }

  // IMPORTANT: call this function before using forwProbs,backProbs
  public void computeParamProbs() {
    if(paramProbsVersion == paramsVersion) return;
    paramProbsVersion = paramsVersion;
    // Forward probabilities: (i,j) => p(j|i)
    if(useForwModel) {
      this.forwProbs = new double[L1.N+1][L2.N];
      for(int i = 0; i <= L1.N; i++) {
        double[] probs = new double[L2.N];
        for(int j = 0; j < L2.N; j++) probs[j] = getScore(i, j);
        NumUtils.expNormalize(probs);
        for(int j = 0; j < L2.N; j++) forwProbs[i][j] = probs[j];
      }
      NumUtils.assertIsFinite(forwProbs);
    }

    // Backward probabilities: (i,j) => p(i|j)
    if(useBackModel) {
      this.backProbs = new double[L1.N][L2.N+1];
      for(int j = 0; j <= L2.N; j++) {
        double[] probs = new double[L1.N];
        for(int i = 0; i < L1.N; i++) probs[i] = getScore(i, j);
        NumUtils.expNormalize(probs);
        for(int i = 0; i < L1.N; i++) backProbs[i][j] = probs[i];
      }
      NumUtils.assertIsFinite(backProbs);
    }
  }

  public void initTruth() {
    for(int i = 0; i < L1.N; i++) {
      for(int j = 0; j < L2.N; j++) {
        if(useForwModel) mu1[i][j] = isTrueTranslation(i, j) ? 1 : 0;
        if(useBackModel) mu2[i][j] = isTrueTranslation(i, j) ? 1 : 0;
      }
    }
  }
  public void initSeed() {
    for(int i = 0; i < L1.N; i++) {
      for(int j = 0; j < L2.N; j++) {
        if(useForwModel) mu1[i][j] = isSeed(i, j) ? 1 : 0;
        if(useBackModel) mu2[i][j] = isSeed(i, j) ? 1 : 0;
      }
    }
  }
  double[] randProbs(int n) {
    double[] probs = new double[n];
    for(int i = 0; i < n; i++)
      probs[i] = 1+opts.initRandom.nextDouble();
    NumUtils.normalize(probs);
    return probs;
  }
  public void initRandom() {
    if(useForwModel) {
      for(int j = 0; j < L2.N; j++) {
        double[] probs = randProbs(L1.N);
        for(int i = 0; i < L1.N; i++)
          mu1[i][j] = probs[i];
      }
    }
    if(useBackModel) {
      for(int i = 0; i < L1.N; i++) {
        double[] probs = randProbs(L2.N);
        for(int j = 0; j < L2.N; j++)
          mu2[i][j] = probs[j];
      }
    }
  }


  public void updateCCARepn(int iter) {
    if(!useCCA()) return;

		track("updateCCARepn");

    track("CCA");
    PairModel model = new ProbCCAModel(DictionaryInducerTester.pccaOptions);
    // Add pairs of words that have the highest weight
    ArrayList<Pair<Pair<double[],double[]>,Double>> muPairs = new ArrayList();
    for(int i = 0; i < L1.N; i++)
      for(int j = 0; j < L2.N; j++)
        muPairs.add(new Pair(new Pair(L1.choleskyRepn[i], L2.choleskyRepn[j]), mu(i, j)));
    Collections.sort(muPairs, new Pair.ReverseSecondComparator());

    ArrayList<Pair<double[],double[]>> pairs = new ArrayList();
    int numTake = (int)Math.min(numSeeds * Math.pow(1.1, iter), muPairs.size());
    for(int i = 0; i < numTake; i++)
      pairs.add(muPairs.get(i).getFirst());
    logs("Taking %d pairs with mu from %s to %s", numTake, muPairs.get(0).getSecond(), muPairs.get(numTake-1).getSecond());
		model.learn(pairs);		
    end_track();

    track("Compute features");
    for(int i = 0; i < L1.N; i++)
      for(int j = 0; j < L2.N; j++)
        biFeatures[i][j].put("cca", model.getScore(L1.choleskyRepn[i], L2.choleskyRepn[j]));
    end_track();
  }

  // Return the entropy that we optimized
  public double eStep() {
    track("E-step");
    computeParamProbs();
    entropy = 0;
    // Forward model (generate j)
    if(useForwModel) {
      for(int j = 0; j < L2.N; j++) {
        double[] probs = new double[L1.N+1];
        for(int i = 0; i <= L1.N; i++)
          probs[i] = forwProbs[i][j];
        if(!opts.allowNull) probs[L1.N] = 0;
        applyTemperatureAndNormalize(probs);
        entropy += NumUtils.entropy(probs);
        for(int i = 0; i <= L1.N; i++) mu1[i][j] = probs[i];
      }
    }

    // Backward model (generate i)
    if(useBackModel) {
      for(int i = 0; i < L1.N; i++) {
        double[] probs = new double[L2.N+1];
        for(int j = 0; j <= L2.N; j++)
          probs[j] = backProbs[i][j];
        if(!opts.allowNull) probs[L2.N] = 0;
        applyTemperatureAndNormalize(probs);
        entropy += NumUtils.entropy(probs);
        for(int j = 0; j <= L2.N; j++) mu2[i][j] = probs[j];
      }
    }
    //logs("entropy = %s", Fmt.D(entropy));
    end_track();
    return entropy;
  }

  void applyTemperatureAndNormalize(double[] probs) {
    if(opts.temperature == 0) {
      double max = ListUtils.max(probs);
      for(int i = 0; i < probs.length; i++)
        probs[i] = Math.abs(probs[i]-max) < 1e-10 ? 1 : 0;
      NumUtils.normalize(probs);
      return;
    }

    for(int i = 0; i < probs.length; i++)
      probs[i] = Math.log(probs[i])/opts.temperature;
    NumUtils.expNormalize(probs);
  }

  public HashMap<String,Double> computeGradient() {
    //track("Compute gradient");
    computeParamProbs();
    HashMap<String,Double> gradient = new HashMap();
    if(useForwModel) {
      for(int j = 0; j < L2.N; j++)
        for(int i = 0; i <= L1.N; i++)
          addToGradient(gradient, mu1[i][j] - forwProbs[i][j], i, j);
    }
    if(useBackModel) {
      for(int i = 0; i < L1.N; i++)
        for(int j = 0; j <= L2.N; j++)
          addToGradient(gradient, mu2[i][j] - backProbs[i][j], i, j);
    }
    // Regularization (assume gradient contains all entries of params)
    for(Map.Entry<String,Double> pair : params.entrySet())
      MapUtils.incr(gradient, pair.getKey(), -opts.regularization * pair.getValue());
    //end_track();
    return gradient;
  }
  void addToGradient(HashMap<String,Double> gradient, double scale, int i, int j) {
    for(Map.Entry<String,Double> pair : biFeatures[i][j].entrySet())
      MapUtils.incr(gradient, pair.getKey(), scale * pair.getValue());
  }

  void addGradientToParams(double scale, HashMap<String,Double> gradient) {
    for(Map.Entry<String,Double> pair : gradient.entrySet())
      MapUtils.incr(params, pair.getKey(), scale * pair.getValue());
    paramsVersion++;
  }

  double computeLogLikelihood() {
    computeParamProbs();
    logLikelihood = 0;
    // Forward model (generate j)
    if(useForwModel) {
      for(int j = 0; j < L2.N; j++) {
        for(int i = 0; i <= L1.N; i++) {
          if(forwProbs[i][j] < 1e-10) {
            assert mu1[i][j] < 1e-10 : mu1[i][j] + " > 0 but prob = " + forwProbs[i][j];
            continue;
          }
          logLikelihood += mu1[i][j] * Math.log(forwProbs[i][j]);
        }
      }
      NumUtils.assertIsFinite(logLikelihood);
    }
    // Backward model (generate i)
    if(useBackModel) {
      for(int i = 0; i < L1.N; i++) {
        for(int j = 0; j <= L2.N; j++) {
          if(backProbs[i][j] < 1e-10) {
            assert mu2[i][j] < 1e-10 : mu2[i][j] + " > 0 but prob = " + backProbs[i][j];
            continue;
          }
          logLikelihood += mu2[i][j] * Math.log(backProbs[i][j]);
        }
      }
      NumUtils.assertIsFinite(logLikelihood);
    }
    // Regularization
    logLikelihood -= opts.regularization/2 * l2Norm(params);
    return logLikelihood;
  }

  double l2Norm(Map<String,Double> map) {
    double sum = 0;
    for(double x : map.values())
      sum += x*x;
    return Math.sqrt(sum);
  }

  // Return if converged
  public boolean mStep(int iter) {
    track("M-step");

    updateCCARepn(iter);

    HashMap<String,Double> gradient = computeGradient();
    dbgs("params: " + params);
    dbgs("gradient: " + gradient);
    if(l2Norm(gradient) < opts.convergenceThreshold) { end_track(); return true; }

    double initValue = computeLogLikelihood();
    double currIncr = 0;
    double targetIncr = stepSize;
    boolean success = false;
    for(int r = 0; r < opts.numReductionSteps; r++) {
      addGradientToParams(targetIncr-currIncr, gradient);
      double targetValue = computeLogLikelihood();
      dbgs("targetIncr = %s, targetValue = %s, initValue = %s", Fmt.D(targetIncr), Fmt.D(targetValue), Fmt.D(initValue));
      if(Math.abs(targetValue-initValue) < opts.convergenceThreshold) { end_track(); return true; }
      if(targetValue > initValue) {
        logs("Increased log-likelihood from %s to %s using step size %s", Fmt.D(initValue), Fmt.D(targetValue), targetIncr);
        stepSize = targetIncr * 1.3; // Next time, try to use a bit larger than this step size
        success = true;
        break;
      }
      currIncr = targetIncr;
      targetIncr *= 0.8;
    }
    if(!success) {
      logs("Failed to increase log-likelihood");
      stepSize = targetIncr;
    }
    end_track();
    return false;
  }

  public void outputParams(String path) {
    if(path == null) return;
    PrintWriter out = IOUtils.openOutHard(path);
    for(Map.Entry<String,Double> pair : params.entrySet())
      out.printf("%s\t%s\n", pair.getKey(), Fmt.D((double)pair.getValue()));
    out.close();
  }

  public void outputPredictions(String path) {
    if(path == null) return;
    PrintWriter out = IOUtils.openOutHard(path);
    int numCorrect = 0;
    double correctSum = 0;
    int numPotentialCorrect = 0;
    // Print out null-aligned?
    for(int i : ListUtils.sortedIndices(L1.freq, true)) {
      boolean isFirst = true;
      double[] mu = new double[L2.N];
      for(int j = 0; j < L2.N; j++) mu[j] = mu(i, j);
      for(int j : ListUtils.sortedIndices(mu, true)) {
        boolean isTrans = isTrueTranslation(i, j);
        boolean isSeed = isSeed(i, j);
        if(isTrans) {
          if(isFirst) numCorrect++;
          correctSum += mu[j];
          numPotentialCorrect++;
        }
        if(mu[j] >= 1e-4 || isTrans)
          out.printf("%s %s\t%s = %s + %s%s%s\n", L1.wstr(i), L2.wstr(j), Fmt.D(mu[j]),
              useForwModel ? Fmt.D(mu1[i][j]) : "_",
              useBackModel ? Fmt.D(mu2[i][j]) : "_",
              isTrans ? (isFirst ? " [CORRECT]" : " [WRONG]") : "", isSeed ? " [SEED]" : "");
          //out.printf("%s %s\t%s%s%s\n", L1.wstr(i), L2.wstr(j), Fmt.D(mu[j]),
              //isTrans ? (isFirst ? " [CORRECT]" : " [WRONG]") : "", isSeed ? " [SEED]" : "");
        isFirst = false;
      }
    }
    logs("Correct: %d/%d = %s; sum of correct mu = %s (%d words total)",
        numCorrect, numPotentialCorrect, Fmt.D(1.0*numCorrect/numPotentialCorrect), correctSum, L1.N);
    out.close();
  }

  public void trainModel() {
    if(useForwModel) this.mu1 = new double[L1.N+1][L2.N];
    if(useBackModel) this.mu2 = new double[L1.N][L2.N+1];
    this.params = new HashMap();
    this.stepSize = opts.initStepSize;

    // Initialize
    switch(opts.initType) {
      case truePred: initTruth(); break;
      case seedPred: initSeed(); break;
      case randomPred: initRandom(); break;
      case editDist: params.put("editDist", -0.01); eStep(); break;
      default: throw Exceptions.unknownCase;
    }

    boolean converged = false;
    for(int iter = 0; iter < opts.numIters && !converged; iter++) {
      track("Iteration %d/%d", iter, opts.numIters);
      if(opts.tmpExample) outputPredictions("/dev/stdout");
      converged = mStep(iter);
      outputParams(Execution.getFile("params."+iter));
      eStep();
      outputPredictions(Execution.getFile("predictions."+iter));
      logs("Objective: T (%s) * entropy (%s) + logLikelihood (%s) = %s",
          Fmt.D(opts.temperature), Fmt.D(entropy), Fmt.D(logLikelihood),
          Fmt.D(opts.temperature*entropy+logLikelihood));
      end_track();
    }
    if(converged) logs("Converged");
  }

	public double[][] getMatchingMatrix(Indexer<String> words1, Indexer<String> words2) {
    extractFeatures();
    trainModel();
		double[][] dists = new double[words1.size()][words2.size()];
    for(int i = 0; i < words1.size(); i++)
      for(int j = 0; j < words2.size(); j++)
        dists[i][j] = mu(L1.words.indexOf(words1.getObject(i)), L2.words.indexOf(words2.getObject(j)));
		return dists;
	}


  // Useless functions
	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
	} public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException("Representations undefined on this inducer.");
	}
	public void setSeedMapping(CounterMap<String, String> seedMapping) { }
}
