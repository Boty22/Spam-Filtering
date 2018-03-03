

"""Reads the data and return a list with the data in the txt"""



def ReadEmail(filename):
    data=[]
    try:
        fh = open(filename,'r')
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


filename = '/home/pili/T2/dataset_1/train/ham/0330.2000-02-04.farmer.ham.txt'

d = ReadEmail(filename)

print(d)
print(d[20])
    #print(data)