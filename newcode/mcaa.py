from common import *
import IO

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
    return X


def mcca(exp_id, X, Y, options):
    X = setupFeatures(options, X)
    Y = setupFeatures(options, Y)
    # X and Y contain features
    print exp_id


if __name__ == '__main__':
    # load data
    fileX = (sys.argv[1])
    fileY = (sys.argv[2])
    X = IO.readFeatures(fileX)
    Y = IO.readFeatures(fileY)
    # set params
    exp_id = 1
    options = Options()
    options.seed_start = 100
    options.weight_type = 'inner'
    mcca(exp_id, X, Y, options)

