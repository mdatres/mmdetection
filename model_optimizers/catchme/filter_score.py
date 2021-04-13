# Filter the prediction according to the score
import csv 
import os

def filter_score(path_to_pred, path_to_save, score_thres):
    ''' Args: - path_to_pred: path to the file with the predictions 
              - path_to_save: path + name to the file where to save the filtered predition
    '''
    try: 
        os.remove(path_to_save)
        d = list()
        with open(path_to_pred, mode='r') as true:
            reader = csv.reader(true)
            for r in reader:
                if float(r[6]) >= score_thres: 
                    d.append(r)

        with open(path_to_save, mode='a') as csvfile:
            csvfile_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for el in d: 
                csvfile_writer.writerow(el)
    except: 
        d = list()
        with open(path_to_pred, mode='r') as true:
            reader = csv.reader(true)
            for r in reader:
                if float(r[6]) >= score_thres: 
                    d.append(r)

        with open(path_to_save, mode='a') as csvfile:
            csvfile_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for el in d: 
                csvfile_writer.writerow(el)