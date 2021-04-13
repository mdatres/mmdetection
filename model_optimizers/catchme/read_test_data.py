import csv
from PIL import Image 

def read_test_data(filepath, patch=False):
    """
    Args: - path of the folder containing the tests data 
    Return: - list containing the unpatch test data's names 
    """
    if patch:
        """
        Args: - path of the folder containing the tests data 
        Return: - list containing all the patch test data path 
        """
        ret = list()
        data_name = list()
        l = list()
        for i in range(0,12): 
            name = "_r_" + str(i) + ".jpg"
            l.append(name)

        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                n = "".join(row[0].split('_r_')[0])
                #path = '/Users/massimilianodatres/'+n[1:]
                path = n
                data_name.append(path)
        data_name = set(data_name) 
        for d in data_name:
            for j in l: 
                ret.append(d+j)
        return(ret)

    else:
        ret = list()
        
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                n = row[0]
                #path = '/Users/massimilianodatres/'+n[1:]
                ret.append(n)
    
        return (list(set(ret)))