from common import *
import IO
import graphs

if __name__ == '__main__':
    wordsFilenameX = (sys.argv[1])
    seedsFilenameX = (sys.argv[2])
    graph_type = (sys.argv[3])

    wordsX = IO.readWords(wordsFilenameX)
    seedsX = IO.readWords(seedsFilenameX)

    concatX = Words.concat(wordsX, seedsX)
    concatX.setupFeatures()
    N, D = concatX.features.shape
    if graph_type.upper() == 'KNN':
        K = int(sys.argv[4])
        (G, I) = graphs.knngraph(concatX.features, K+1)
        G = G - np.eye(N)

        graphFilename = wordsFilenameX.replace(".", "_graph.")
        G = np.mat(G)
        print np.sum(G, 1)
        IO.writeNumpyArray(graphFilename, G)