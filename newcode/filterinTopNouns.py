import IO
import sys
import perm


def filter(X, N):
    (freq, I) = perm.sort(X.freq, reverse=True)
    I = I[:N]
    X.features = X.features[I, :]
    X.words = X.words[I]
    X.freq = X.freq[I]
    # filter in the top N words
    return X

if __name__ == '__main__':
    # filters in the top N frequent words/freq/features and writes them to a file
    fileX = (sys.argv[1])
    N = int(sys.argv[2])
    X = IO.readWords(fileX)
    X = filter(X, N)
    outname = fileX.replace(".", "_N=" + str(N) + ".")
    IO.writeWords(outname, X)
