from common import *
import IO


def find_matching(options, X, Y):
    # finds a permutation pi that best matches Y to X
    # TODO: write the code
    print 1


def setupFeatures(options, X):
    (N, D) = X.features.shape
    (freq, I) = perm.sort(X.freq, reverse=True)
    I = I[:options.max_words]
    X.features = X.features[I, :]
    X.words = X.words[I]
    logFr = np.log(X.freq)
    L = strings.strlen(X.words)
    X.features = normalize_rows(X.features)
    # TODO: should be add logFr and L ?
    return X


# this method permutes all the fields of X according to pi.
# if pi is shorter than X.words, than only the first entries are permuted,
# and the last remain in their poisition.
def permuteFirstWords(X, pi):
    id = np.arange(len(pi))
    X.words[id] = X.words[pi]
    X.features[id, :] = X.features[pi, :]
    #X.G[id,id] = X.G[pi, pi]
    return X


# returns the permutation that sorts edge_cost, but maps the smallest K elements in edge_cost to the end.
# e.g. if edge_cost is [1,2,10,1,2] and K is 2
# then I should sort it to [2,2,10,1,1]
def buildPerm(K, edge_cost):
    (sorted_ed_edge_cost, I) = perm.sort(edge_cost)
    Iseed = I[:K].tolist()
    Irest = I[K:].tolist()
    I = np.array(Irest + Iseed)
    return I


def mcca(X, Y, options):
    X = setupFeatures(options, X)
    Y = setupFeatures(options, Y)

    edit_dist_options = Options()
    edit_dist_options.exp_id = -1  # Edit distance exp_id is -1.
    (ed_pi, ed_edge_cost) = IO.readMatching(edit_dist_options, X.words, Y.words)
    # TODO: continue from this point
    # You need to setup the initial matching according to ed_pi
    I = buildPerm(options, ed_edge_cost)
    X = permuteFirstWords(X, I)
    Y = permuteFirstWords(Y, ed_pi[I])

    return (pi, edge_cost)

if __name__ == '__main__':
    np.random.seed(1)
    # load data
    #fileX = (sys.argv[1])
    #fileY = (sys.argv[2])
    fileX = '../SCRIPTS/matlab/Jun10_en.txt'
    fileY = '../SCRIPTS/matlab/Jun10_es.txt'
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)
    # set params

    options = Options()
    options.exp_id = 1000
    options.seed_start = 100
    options.max_words = 2000+options.seed_start
    options.weight_type = 'inner'

    (pi, edge_cost) = mcca(X, Y, options)

