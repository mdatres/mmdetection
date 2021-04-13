import csv
from IoU import calc_iou
import numpy as np

# Create a dictionary that contains the prediction and the true-bbbox 

def gt_splitter(path_to_data): 
    data = list()
    with open(path_to_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row)

    scaf_dic = dict()
    plan_dic = dict()
    for d in data:
        if d[5]=='scafoideus_titanus':
            name = d[0].split('/')[-1]
            if name not in scaf_dic.keys(): 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                scaf_dic[name] = [box] 
            else: 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                scaf_dic[name].append(box)
        elif d[5]=='planococcus_ficus_m':
            name = d[0].split('/')[-1]
            if name not in plan_dic.keys(): 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                plan_dic[name] = [box] 
            else: 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                plan_dic[name].append(box) 
        
    return (scaf_dic,plan_dic)

def pred_splitter(path_to_data): 
    data = list()
    with open(path_to_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row)

    scaf_dic = dict()
    plan_dic = dict()
    for d in data:
        if d[5]=='scafoideus_titanus':
            name = d[0].split('/')[-1]
            if name not in scaf_dic.keys(): 
                add = dict()
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                add['boxes'] = [box] 
                add['scores'] = [float(d[6])]
                scaf_dic[name] = add
            else: 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                scaf_dic[name]['boxes'].append(box)
                scaf_dic[name]['scores'].append(float(d[6]))
        
        elif d[5]=='planococcus_ficus_m':
            name = d[0].split('/')[-1]
            if name not in plan_dic.keys(): 
                add = dict()
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                add['boxes'] = [box] 
                add['scores'] = [float(d[6])]
                plan_dic[name] = add
            else: 
                box = [float(d[1]),float(d[2]),float(d[3]),float(d[4])]
                plan_dic[name]['boxes'].append(box)
                plan_dic[name]['scores'].append(float(d[6]))
    
    return (scaf_dic,plan_dic)

# Makes a function that reads all the unpatch files;
def complete_test_images(dic, froms,  pred=True): 
    ret = dic.copy()
    
    intest = list(froms.keys())
    intest = set(intest) 
    for el in intest:  
            if el not in list(dic.keys()): 
                if pred:
                    add = dict()
                    add['boxes'] = []
                    add['score'] = []
                    ret[el] = add
                else: 
                    ret[el] = []
    return ret

# Makes a function that reads all the patch files;
def complete_patch_test_images(dic, froms,  pred=True): 
    
    ret = dic.copy()
    l = list()
    for i in range(0,12): 
        name = "_r_" + str(i) + ".jpg"
        l.append(name)
    intest = list(froms.keys())
    intest = ["".join(i.split('_r_')[0]) for i in intest]
    intest = set(intest) 
    for el in intest: 
        for patch in l: 
            name = el + patch
            if name not in list(dic.keys()): 
                if pred:
                    add = dict()
                    add['boxes'] = []
                    add['score'] = []
                    ret[name] = add
                else: 
                    ret[name] = []
    return ret


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    pred_boxes = pred_boxes['boxes']
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))

    if len(all_pred_indices)==0 and len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=len(all_gt_indices)
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    if len(all_gt_indices)==0:
        tp=0
        fp=len(all_pred_indices)
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=len(all_pred_indices)
        fn=len(all_gt_indices)
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
    
    
    fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


        
def TP_FP_FN(path_to_true, path_to_pred, iou_thr, patch=False):
    '''
    Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args: - path to the file with the annotation 
          - path to the file with the prediction 
    Return:  dict: true positives (int), false positives (int), false negatives (int)
    '''
    scaf_pred, plan_pred = pred_splitter(path_to_pred)
    scaf_gt, plan_gt = gt_splitter(path_to_true)
    
    if patch:
        scaf_gt = complete_patch_test_images(scaf_gt, scaf_gt,  False)
        scaf_gt = complete_patch_test_images(scaf_gt, scaf_pred,  False)
        plan_gt = complete_patch_test_images(plan_gt, plan_gt, False)
        plan_gt = complete_patch_test_images(plan_gt, plan_pred, False)
        scaf_pred = complete_patch_test_images(scaf_pred,scaf_pred)
        scaf_pred = complete_patch_test_images(scaf_pred,scaf_gt)
        plan_pred = complete_patch_test_images(plan_pred,plan_pred)
        plan_pred = complete_patch_test_images(plan_pred,plan_gt)
    else:
        scaf_gt = complete_test_images(scaf_gt, scaf_gt,  False)
        scaf_gt = complete_test_images(scaf_gt, scaf_pred,  False)
        plan_gt = complete_test_images(plan_gt, plan_gt, False)
        plan_gt = complete_test_images(plan_gt, plan_pred, False)
        scaf_pred = complete_test_images(scaf_pred,scaf_pred)
        scaf_pred = complete_test_images(scaf_pred,scaf_gt)
        plan_pred = complete_test_images(plan_pred,plan_pred)
        plan_pred = complete_test_images(plan_pred,plan_gt)
  
    
        
    tp = 0 
    fp = 0 
    fn = 0 
    for el in scaf_gt.keys(): 
        ret = get_single_image_results(scaf_gt[el], scaf_pred[el],iou_thr)
        tp += ret['true_positive']
        fp += ret['false_positive']
        fn += ret['false_negative']


    
    for el in plan_gt.keys():
        ret = get_single_image_results(plan_gt[el], plan_pred[el],iou_thr)
        tp += ret['true_positive']
        fp += ret['false_positive']
        fn += ret['false_negative']
        
    return  {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    
    
    