# -*- coding: utf-8 -*-
"""

@author: Virti Bipin Sanghavi
"""

import ast
# import csv
# import sys
# import math
import os
import csv
import math
import random
from random import randrange
from sklearn.datasets import load_iris


class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):
    freq = {}
    index = attributes.index(target)  # 2 or 3 or 4 or 1
    # print("index", index)
    for tuple in data:
        if (tuple[index] in freq):
            freq[tuple[index]] += 1
        else:
            freq[tuple[index]] = 1
    # print("freq",freq)
    max = 0
    major = ""
    for key in freq.keys():
        if freq[key] > max:
            max = freq[key]
            major = key
    # print("major",major)    #0 or 1 or 2
    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):
    freq = {}
    dataEntropy = 0.0
    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1
    i = i - 1

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for freq in freq.values():
        dataEntropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):
    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for val in freq.keys():
        valProb = freq[val] / sum(freq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):
    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr
    # print("best", best)     #SepalWidth or PetalWidth or PetalLength or SepalLength
    return best


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):
    index = attributes.index(attr)
    # print("index", index)  #index of the attribute - 0 or 1 or 2 or 3
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    # print("values", values)
    return values


# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):
    new_data = [[]]
    index = attributes.index(best)
    # print("index", index)       # 0 or 1 or 2 or 3
    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0, len(entry)):
                if (i != index):
                    newEntry.append(entry[i])
                    # print(newEntry)
            new_data.append(newEntry)
            # print(new_data)

    new_data.remove([])
    # print("new_data:", new_data)
    return new_data


# This function is used to build the decision tree using the given data, attributes and the target attributes. It returns the decision tree in the end.
def build_tree(data, attributes, target):
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    # print("vals:",vals)
    default = majorClass(attributes, data, target)
    # print(vals.count(vals[0]), len(vals), vals[0])
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best: {}}
        # print({best:{}})

        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            # print("new_data:", new_data)
            newAttr = attributes[:]
            # print("newAttr:", newAttr)
            newAttr.remove(best)
            # print("newAttr:", newAttr)
            """
            new_data: [['5.0', '3.5', '1.6', '0']]
            newAttr: ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
            newAttr: ['SepalLength', 'SepalWidth', 'PetalLength', 'Name']            
            """
            subtree = build_tree(new_data, newAttr, target)
            # print("best:", best)        #petalwidth
            # print("val:", val)          #0.6
            tree[best][val] = subtree
    """
    print(tree)
    {'PetalWidth': {'0.6': '0', '0.4': '0', '2.1': '2', '0.5': '0', '2.3': '2', '1.5': 
    {'SepalLength': {'6.7': '1', '6.5': '1', '6.0': '2', '6.4': '1', '6.3': '1', '5.4': '1', '5.9': '1', '6.2': '1', '5.6': '1'}}, 
    '2.4': '2', '1.3': '1', '2.5': '2', '1.7': '1', '1.1': '1', '1.0': '1', '0.2': '0', '1.4': '1', '1.8': 
    {'SepalLength': {'6.7': '2', '6.2': '2', '7.2': '2', '6.0': '2', '7.3': '2', '6.3': '2', '5.9': '1', '6.1': '2', '6.5': '2'}}, '0.3': '0', '1.9': '2', '1.6':
    {'SepalWidth': {'3.3': '1', '3.0': '2', '3.4': '1', '2.7': '1'}}, '2.2': '2', '2.0': '2', '0.1': '0', '1.2': '1'}}
    """
    # print(tree)
    return tree


def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# This function runs the decision tree algorithm. It parses the file for the data-set, and then it runs the 10-fold cross-validation. It also classifies a test-instance and later compute the average accurracy
def run_decision_tree():
    data = []
    with open("iris.txt") as tsv:
        for line in csv.reader(tsv, delimiter=","):

            if line[4] == 'Iris-setosa':
                line[4] = '0'
            elif line[4] == 'Iris-versicolor':
                line[4] = '1'
            else:  # Iris-virginica
                line[4] = '2'
            data.append(tuple(line))

    print("Number of records: %d" % len(data))
    # print(data)

    attributes = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
    target = attributes[-1]

    K = 5
    acc = []
    for k in range(K):
        random.shuffle(data)
        training_set = [x for i, x in enumerate(data) if i % K != k]
        test_set = [x for i, x in enumerate(data) if i % K == k]
        # print(int(len(training_set)))
        # print(int(len(test_set)))
        tree = DecisionTree()
        tree.learn(training_set, attributes, target)
        """
        tree:
    {'PetalWidth': {'0.6': '0', '0.4': '0', '2.1': '2', '0.5': '0', '2.3': '2', '1.5': 
    {'SepalLength': {'6.7': '1', '6.5': '1', '6.0': '2', '6.4': '1', '6.3': '1', '5.4': '1', '5.9': '1', '6.2': '1', '5.6': '1'}}, '2.4': '2', '1.3': '1', '2.5': '2', '1.7': '1', '1.1': '1', '1.0': '1', '0.2': '0', '1.4': '1', '1.8': 
    {'SepalLength': {'6.7': '2', '6.2': '2', '7.2': '2', '6.0': '2', '7.3': '2', '6.3': '2', '5.9': '1', '6.1': '2', '6.5': '2'}}, '0.3': '0', '1.9': '2', '1.6':
    {'SepalWidth': {'3.3': '1', '3.0': '2', '3.4': '1', '2.7': '1'}}, '2.2': '2', '2.0': '2', '0.1': '0', '1.2': '1'}}
        """
        results = []

        for entry in test_set:
            tempDict = tree.tree.copy()
            # print(tempDict)
            result = ""
            while (isinstance(tempDict, dict)):
                root = Node(list(tempDict)[0], tempDict[list(tempDict)[0]])  # creates a Node
                tempDict = tempDict[list(tempDict)[0]]
                """
                print(tempDict[list(tempDict)[0]])
                {'PetalLength': {'3.9': '1', '4.7': '1', '4.4': '1', '4.6': '1', '5.6': '2'}}                
                """
                index = attributes.index(root.value)
                # print("index:", index)  #0 or 1 or 2 or 3
                value = entry[index]
                # print(value)        #1.4
                # print(tempDict.keys())
                # dict_keys(['6.7', '6.5', '6.3', '6.9', '5.9', '5.4', '5.6'])
                if (value in tempDict.keys()):
                    Node(value, tempDict[value])
                    # print(value, tempDict[value])
                    # 4.8,  {'SepalLength': {'5.9': '1', '6.0': '2'}}
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break
            if result != "Null":
                results.append(result == entry[-1])
            # print("results: ", results)
            """
            results:  [True]
            results:  [True]
            results:  [True, True]
            results:  [True, True]
            results:  [True, True, True]
            results:  [True, True, True, True]
            results:  [True, True, True, True]
            results:  [True, True, True, True, True]
            results:  [True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True]
            results:  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True]
            """
        accuracy = float(results.count(True)) / float(len(results))
        # print("accuracy:", accuracy)
        acc.append(accuracy)

    avg_acc = sum(acc) / len(acc)
    print("Average accuracy: %.4f" % avg_acc)

    # Writing results to a file (DO NOT CHANGE)
    f = open("result.txt", "w")
    f.write("accuracy: %.4f" % avg_acc)
    f.close()

#
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# iris = load_iris()
# # print(iris.data)
# # print(iris.target)
# clf = clf.fit(iris.data, iris.target)
# print(clf)
# dot_data = StringIO()
# print(tree.export_graphviz(clf, out_file=dot_data))
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")
# import os

#os.startfile("iris.pdf")

def load_csv_to_header_data(filename):
    path = os.path.normpath(os.getcwd() + filename)
    fs = csv.reader(open(path))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    return data


def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def project_columns(data, columns_to_project):
    data_h = list(data['header'])
    data_r = list(data['rows'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h, 'rows': data_r,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name}


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map


def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def entropy1(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent


def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions


def avg_entropy_w_partitions(data, splitting_att, target_attribute):
    # find uniq values of splitting att
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target_attribute)
        partition_entropy = entropy1(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, uniqs, remaining_atts, target_attribute):
    labels = get_class_labels(data, target_attribute)

    node = {}

    if len(labels.keys()) == 1:
        node['label'] = next(iter(labels.keys()))
        return node

    if len(remaining_atts) == 0:
        node['label'] = most_common_label(labels)
        return node

    n = len(data['rows'])
    ent = entropy1(n, labels)

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    for remaining_att in remaining_atts:
        avg_ent, partitions = avg_entropy_w_partitions(data, remaining_att, target_attribute)
        info_gain = ent - avg_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = remaining_att
            max_info_gain_partitions = partitions

    if max_info_gain is None:
        node['label'] = most_common_label(labels)
        return node

    node['attribute'] = max_info_gain_att
    node['nodes'] = {}

    remaining_atts_for_subtrees = set(remaining_atts)
    remaining_atts_for_subtrees.discard(max_info_gain_att)

    uniq_att_values = uniqs[max_info_gain_att]

    for att_value in uniq_att_values:
        if att_value not in max_info_gain_partitions.keys():
            node['nodes'][att_value] = {'label': most_common_label(labels)}
            continue
        partition = max_info_gain_partitions[att_value]
        node['nodes'][att_value] = id3(partition, uniqs, remaining_atts_for_subtrees, target_attribute)

    return node


def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    return ast.literal_eval(data)


def pretty_print_tree(root):
    stack = []
    rules = set()

    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.add(''.join(stack))
            stack.pop()
        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')
            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()
            stack.pop()

    traverse(root, stack, rules)
    print(os.linesep.join(rules))


def main():
    #argv = sys.argv
    #print("Command line args are {}: ".format(argv))

    config = load_config("iris.cfg")

    data = load_csv_to_header_data(config['data_file'])
    data = project_columns(data, config['data_project_columns'])

    target_attribute = config['target_attribute']
    remaining_attributes = set(data['header'])
    remaining_attributes.remove(target_attribute)

    uniqs = get_uniq_values(data)

    root = id3(data, uniqs, remaining_attributes, target_attribute)

    pretty_print_tree(root)


if __name__ == "__main__":
    main()
    run_decision_tree()