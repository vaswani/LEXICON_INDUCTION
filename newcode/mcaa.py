from common import *
import IO


def find_matching(options, X, Y):
    # finds a permutation pi that best matches Y to X
    # TODO: write the code

def setupFeatures(options, X):
    (N,D) = X.features.shape
    seedlen = options.seed_start;
    (freq, I) = perm.sort(X.freq)
    I = I[:seedlen-1]
    X.features = X.features[I, :]
    X.words = X.words[I]
    logFr = np.log(X.freq)
    L = strings.strlen(X.words)
    X.features = normalize_rows(X.features)
    # TODO: should be add logFr and L ?
    return X


def mcca(X, Y, options):
    X = setupFeatures(options, X)
    Y = setupFeatures(options, Y)

    edit_dist_options = Options();
    edit_dist_options.exp_id = -1;
    (pi, edge_cost) = IO.readMatching(edit_dist_options, X.words, Y.words)
    # TODO: continue from this point
    return (pi, edge_cost)

if __name__ == '__main__':
    # load data
    fileX = (sys.argv[1])
    fileY = (sys.argv[2])
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)
    # set params

    options = Options()
    options.exp_id = 1000
    options.seed_start = 100
    options.weight_type = 'inner'

    mcca(X, Y, options)

