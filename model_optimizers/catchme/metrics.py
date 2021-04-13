from TP_FP_FN import TP_FP_FN
import csv

def CSI(path_to_true, path_to_pred, iou, patch=False): 
    ''' Args:  - path to the file with the annotation 
               - path to the file with the prediction 
               - iou used in computing TP_FP_FN
        Return:  The value of the CSI on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']
    return tp/(tp + fp + fn)

def FAR(path_to_true, path_to_pred, iou, patch=False): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction 
              - iou used in computing TP_FP_FN
        Return:  The value of the False Alarm Rate(FAR) on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']


    return fp/(tp + fp)

def postrate(path_to_true, path_to_pred, iou):
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, False) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']
    gr_count = 0 
    with open(path_to_true) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            gr_count +=1

    return (tp+fn)/gr_count   

def PREC(path_to_true, path_to_pred, iou, patch=False): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction 
              - iou used in computing TP_FP_FN
        Return:  The value of the PRECISION on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']

    try:
        return tp/(tp + fp)
    except: 
        return 0

def REC(path_to_true, path_to_pred, iou, patch=False): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction 
              - iou used in computing TP_FP_FN
        Return:  The value of the RECALL on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']


    return tp/(tp + fn)    


def POD(path_to_true, path_to_pred, iou, patch=False): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction
              - iou used in computing TP_FP_FN 
        Return:  The value of the Probability of Detection/Recall(POD) on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']

    return tp/(tp + fn)

def SR(path_to_true, path_to_pred, iou, patch=False): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction 
              - iou used in computing TP_FP_FN
        Return:  The value of the Precision(SR) on the entire test set
    '''
    dic = TP_FP_FN(path_to_true, path_to_pred, iou, patch) 
    tp = dic['true_positive']
    fp = dic['false_positive']
    fn = dic['false_negative']

    return tp/(tp + fp)

def countersPercentage(path_to_true, path_to_pred): 
    '''Args:  - path to the file with the annotation 
              - path to the file with the prediction 
        Return:  The value of the percentage of counter on the entire test set
    '''
    
    pred_count_scaf = 0 
    pred_count_ficus = 0 
    with open(path_to_pred) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[5]=='scafoideus_titanus': 
                pred_count_scaf +=1
            elif row[5]=='planococcus_ficus_m': 
                pred_count_ficus +=1
    
    
    gr_count_scaf = 0 
    gr_count_ficus = 0 
    with open(path_to_true) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[5]=='scafoideus_titanus': 
                gr_count_scaf +=1
            elif row[5]=='planococcus_ficus_m': 
                gr_count_ficus +=1
    
    return {'scafoideus_titanus': pred_count_scaf/gr_count_scaf,'planococcus_ficus_m': pred_count_ficus/gr_count_ficus}
            


