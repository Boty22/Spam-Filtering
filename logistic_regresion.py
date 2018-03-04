import os
#import pandas as pd

def readEmail(filename):
    """input: name of a the txt file
        output: list of words of the message in the email
    """
    #data = [filename]
    data = []
    try:
        fh = open(filename,'r',encoding = 'latin1')
    except IOError:
        print('cannot open', filename)
    else:
        for line in fh:
            if line !='\n':
                words =  line[:-1].split(' ')
                for word in words:
                    data.append(word)
    finally:
        fh.close()
    return data

def readDirectory(directory, class_name):
    '''Takes the name of the directory and the name of the class
        and creates a subset of data  
    '''
    d = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            d.append([readEmail(os.path.join(root,name)),class_name])
    return d

def extractVocabulary(samples):
    """
    """
    vocabulary = []
    for sample in samples:
        #print (sample)
        for word in sample[0]:
            #print(word)
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

def split_70_30(samples):
    total_samples = len(samples)
    split_limit = int(total_samples*7/10)
    f = []
    l = []
    counter = 0
    for sample in samples:
        if counter <= split_limit:
            f.append(sample)
            counter += 1
        else:
            l.append(sample)
            counter += 1
    return f,l 

def filter_data_by_class(samples,class_name):
    result = []
    for sample in samples:
        if sample[1] == class_name:
            result.append(sample)
    return result

def split_training_data(classes_names,samples):
    first70 = []
    last30 = []
    for class_name in classes_names:
        filtered_data_by_class = filter_data_by_class(samples,class_name)
        print(len(filtered_data_by_class))
        f,l = split_70_30(filtered_data_by_class)
        first70 = first70 + f
        last30 = last30 + l
    return first70,last30


""" ------------------ Testing functionality ------------------ """

train_ham_file = '/home/pili/T2/dataset_1/train/ham'
train_spam_file = '/home/pili/T2/dataset_1/train/spam'
test_ham_file = '/home/pili/T2/dataset_1/test/ham'
test_spam_file = '/home/pili/T2/dataset_1/test/spam'

data_train = readDirectory(train_spam_file,1) + readDirectory(train_ham_file,0) 
C = [0,1]

samples_70_train,samples_30_validation = split_training_data(C,data_train)