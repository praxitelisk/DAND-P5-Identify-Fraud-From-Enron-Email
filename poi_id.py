#!/usr/bin/python

#######################
# This python script is just the tip of the iceberg and a starting point.
# For a full walkthrough of the process follow the following link:
# https://github.com/praxitelisk/NDDA-P5-Identify-Fraud-From-Enron-Email
#######################

import sys
import pickle
from tester import dump_classifier_and_data
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'exercised_stock_options', 'fraction_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

#remove datapoints that are noisy
data_dict.pop('FREVERT MARK A',0)
data_dict.pop('LAVORATO JOHN J',0)
data_dict.pop('BUY RICHARD B',0)
data_dict.pop('BAXTER JOHN C',0)
data_dict.pop('HAEDICKE MARK E',0)
data_dict.pop('KEAN STEVEN J',0)
data_dict.pop('WHALLEY LAWRENCE G',0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages


    return fraction

for name in my_dataset:

    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    my_dataset[name]["fraction_to_poi"] = fraction_to_poi

    
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variaty of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import StratifiedShuffleSplit

X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
for train_index, test_index in sss:
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test = y[train_index], y[test_index]
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


def dump_classifier_and_data(clf, dataset, feature_list):

    CLF_PICKLE_FILENAME = "my_classifier.pkl"
    DATASET_PICKLE_FILENAME = "my_dataset.pkl"
    FEATURE_LIST_FILENAME = "my_feature_list.pkl"

    with open(CLF_PICKLE_FILENAME, 'wb') as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, 'wb') as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, 'wb') as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

dump_classifier_and_data(clf, my_dataset, features_list)