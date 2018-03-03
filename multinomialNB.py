#The first step is to walk among all directories and file
import os

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

def extraactVocabulary(samples):
    """
    """
    vocabulary = []
    for sample in samples:
        for word in sample[1]:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

def countDocs(samples):
    return len(samples)

def countDocsInClass(samples,class_name):
    count = 0
    for sample in samples:
        if sample[1] == class_name:
            count += 1
    return count

def concatenateTextOfAllDocsInClass(samples,class_name):
    text_concatenated = []
    for sample in samples:
        if sample[1] == class_name:
            for word in sample[0]:
                text_concatenated.append(word)
    return text_concatenated

def countTokesnOfTerm(text,t):
    tokens = 0
    for word in text:
        if word == t:
            tokens += 1
    return tokens





fsamples = [[1,['lucia','botiquin','ortiz']],[1,['nataly','botiquin','ortiz']],[0,['julio','cesar','botiquin']]]
fv = extraactVocabulary(fsamples)
print(fv)

#the name of the files I want to reare are
train_ham_file = '/home/pili/T2/dataset_1/train/ham'
train_spam_file = '/home/pili/T2/dataset_1/train/spam'
test_ham_file = '/home/pili/T2/dataset_1/test/ham'
test_spam_file = '/home/pili/T2/dataset_1/test/spam'


data = readDirectory(train_spam_file,1) + readDirectory(train_ham_file,0) 
print(len(data))
print(countDocsInClass(data,0))
print(len(concatenateTextOfAllDocsInClass(data,0)))
ftext = concatenateTextOfAllDocsInClass(data,0)
print(countTokesnOfTerm(ftext,'love'))

