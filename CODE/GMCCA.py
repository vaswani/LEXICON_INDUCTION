import sys
import numpy
from Util import *
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("--source_feature_file", action="store", type="string", dest="source_feature_file",default='train.txt',help="The path to the source feature file")
    parser.add_option("--source_graph_file", action="store", type="string", dest="source_graph_file",default='train.txt',help="The path to the source feature file")
    parser.add_option("--target_feature_file", action="store", type="string", dest="target_feature_file",default='train.txt',help="The path to the target feature file")
    parser.add_option("--target_graph_file", action="store", type="string", dest="target_graph_file",default='train.txt',help="The path to the target feature file")

    parser.add_option("--eta", action="store", type="float", dest="eta",default ='0.',help="The weight of the graph ")

    '''
    parser.add_option("--test_file", action="store", type="string", dest="test_file",default ='test.txt',help="The test file ")

    parser.add_option("--n_hidden", action="store", type="int", dest="n_hidden",default =500,help="Number of nodes in the hidden layer. Default is 500")
    
    parser.add_option("--n_out", action="store", type="int", dest="n_out",default =7,help="Number of nodes in the output softmax layer. Default is 7")
    parser.add_option("--batch_size", action="store", type="int", dest="batch_size",default =20,help="batch_size. Default is 20")
    parser.add_option("--n_epochs", action="store", type="int", dest="n_epochs",default =1000,help="number of epochs. Default is 1000")

    parser.add_option("--L1_reg", action="store", type="float", dest="L1_reg",default =0.0,help="L1 reg penalty. Default is 0.0 ")
    parser.add_option("--L2_reg", action="store", type="float", dest="L2_reg",default =0.0001,help="L2 reg penalty. Default is 0.0001 ")

    parser.add_option("--learning_rate", action="store", type="float", dest="learning_rate",default =0.02,help="Learning Rate. Default is 0.02 ")
    
    
    parser.add_option("--activation", action="store", type="string", dest="activation",default ='TANH',help="SIGMOID or TANH activaiton. Default is TANH")
    '''
    (options, args) = parser.parse_args()
    
    #print options
    #print args
    
    #getting the values from optparse
    source_feature_file = options.source_feature_file
    source_graph_file = options.source_graph_file
    target_feature_file = options.target_feature_file
    target_graph_file = options.target_graph_file

    eta = options.eta
    '''
    valid_file= options.valid_file
    test_file = options.test_file
    n_hidden = options.n_hidden
    n_out = options.n_out
    batch_size = options.batch_size
    n_epochs = options.n_epochs
    L1_reg = options.L1_reg
    L2_reg = options.L2_reg
    activation = options.activation
    learning_rate = options.learning_rate
    '''

    #read in source features
    source_index_to_word,source_features = readFeatures(source_feature_file)
    print '...read source context features'
    source_word_to_index = dict((word,i)for i,word in enumerate(source_index_to_word))
    source_graph = readGraph(source_graph_file,source_word_to_index)
    print '...read source graph'
    print source_word_to_index

    #read in target features
    target_index_to_word,target_features = readFeatures(target_feature_file)
    print '...read target context features'
    target_word_to_index = dict((word,i)for i,word in enumerate(target_index_to_word))
    target_graph = readGraph(target_graph_file,target_word_to_index)
    print '...read target graph'
    print target_word_to_index

    #print source_index_to_word 
    print source_features

    #print source_index_to_word 
    print source_features
    print eta
  


if __name__== "__main__":
    main()


