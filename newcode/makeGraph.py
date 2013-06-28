from words import *
import IO
import graphs


def makeGraph(wordsX, seedsX, graph_type, K):
    concatX = Words.concat(wordsX, seedsX)
    #concatX.setupFeatures() # when using mock data, normalization is sensitive to mean shifting.
    N, D = concatX.features.shape
    if graph_type.upper() == 'KNN':
        (G, I) = graphs.knngraph(concatX.features, K+1)
        G = G - np.eye(N)  # remove self
        G = np.mat(G)
    return G

if __name__ == '__main__':
    # parse arguments
    filename_wordsX = (sys.argv[1])
    filename_seedX = (sys.argv[2])
    graph_type = (sys.argv[3])
    K = int(sys.argv[4])

    # parse files
    wordsX = IO.readWords(filename_wordsX)
    seedsX = IO.readWords(filename_seedX)
    graphFilename = filename_wordsX.replace(".", "_graph.")

    # make graph
    G = makeGraph(wordsX, seedsX, graph_type)

    print np.sum(G, 1)
    # save graph
    IO.writeNumpyArray(graphFilename, G)