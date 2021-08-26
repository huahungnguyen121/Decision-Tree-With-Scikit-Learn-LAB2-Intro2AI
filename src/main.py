from utility import *

X, y = readInputFile() # read mushrooms data set to X, y with y is the target attribute
subsets = prepDataset(X, y) # prepare training sets and test sets of 4 different proportions
trees = constructDecisionTree(subsets) # build the decision trees for each training set
printOutAllReports(trees, subsets) # print out all reports about prediction using those decision trees
constructDecisionTreeWithDepth(subsets) # build the decision trees with different depths and print out the predictive reports for 7 depths of those trees