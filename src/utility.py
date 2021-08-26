import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

def readInputFile():
    df = pd.read_csv('D:\Homework\Artificial Intelligent\Lab2\mushrooms.csv', sep=',') # read mushrooms.csv to dataframe
    data = df.drop('class', axis=1).copy()
    X = pd.get_dummies(data, columns=data.columns.values.tolist()).copy()

    res = df['class'].copy()
    res[res == 'p'] = 1 # change p to 1
    res[res == 'e'] = 0 # change e to 0
    y = res.values.tolist()

    return X, y

def prepDataset(X, y):
    ratio = [0.4, 0.6, 0.8, 0.9] # list of ratios (train/total) using for prepare the subsets
    subsetItems = []

    for r in ratio:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=r)
        subsetItems.append({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})
    
    return subsetItems # list of objects which contain train sets and test sets for each ratio above


def constructDecisionTree(trainTestSubsets):
    split_ratio = [0.4, 0.6, 0.8, 0.9]
    decisionTrees = []

    for subset, ratio in zip(trainTestSubsets, split_ratio):
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(subset['X_train'], subset['y_train'])
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=subset['X_train'].columns, class_names=['e','p'], rounded=True, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render('output/train_{}_decision_tree'.format(ratio), cleanup=True) #file's name and file's path for exporting
        decisionTrees.append(clf)
    
    return decisionTrees # list of decision trees

def printOutAllReports(decisionTrees, testSets):
    for tree, test, i in zip(decisionTrees, testSets, range(1, len(decisionTrees) + 1)):
        y_pred = tree.predict(test['X_test'])
        print("Report #{}: \n".format(i), classification_report(test['y_test'], y_pred, target_names=['e','p']))
        plot_confusion_matrix(tree, test['X_test'], test['y_test'])
        filename = 'output/confusion_matrix_{}.png'.format(i) #file's name and file's path for exporting
        print('Confusion matrix was saved as {}\n'.format(filename))
        plt.savefig(filename)
        plt.clf()
    return

def constructDecisionTreeWithDepth(subsets):
    depthList = [None, 2, 3, 4, 5, 6, 7]
    for depth in depthList:
        # build dicision trees of different depths
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        clf = clf.fit(subsets[2]['X_train'], subsets[2]['y_train'])
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=subsets[2]['X_train'].columns, class_names=['e','p'], rounded=True, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render('output/decision_tree_with_depth_{}'.format(depth), cleanup=True) #file's name and file's path for exporting

        # print out classification report and confusion matrix for each tree with corresponding test sets
        y_pred = clf.predict(subsets[2]['X_test'])
        print("Report (depth = {}): \n".format(depth), classification_report(subsets[2]['y_test'], y_pred, target_names=['e','p']))
        plot_confusion_matrix(clf, subsets[2]['X_test'], subsets[2]['y_test'])
        filename = 'output/confusion_matrix_depth_{}.png'.format(depth) #file's name and file's path for exporting
        print('Confusion matrix was saved as {}\n'.format(filename))
        plt.savefig(filename)
        plt.clf()
    return