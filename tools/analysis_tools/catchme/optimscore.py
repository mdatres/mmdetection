import csv
import json
import numpy as np
import argparse
import os 
#from COCO_mAP50_catchme import calc_iou, gt_splitter, pred_splitter,complete_test_images, get_single_image_results, TP_FP_FN, perc_of_predict, go_noARGPARSE

def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_bottomleft_gt, y_bottomleft_gt, x_topright_gt, y_topright_gt= gt_bbox
    x_bottomleft_p, y_bottomleft_p, x_topright_p, y_topright_p= pred_bbox
    
    if (x_bottomleft_gt > x_topright_gt) or (y_bottomleft_gt> y_topright_gt):
        print(x_bottomleft_gt, y_bottomleft_gt, x_topright_gt, y_topright_gt)
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_bottomleft_p > x_topright_p) or (y_bottomleft_p> y_topright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_bottomleft_p, x_topright_p,y_bottomleft_p,y_topright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_topright_gt< x_bottomleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_topright_gt< y_bottomleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_bottomleft_gt> x_topright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_bottomleft_gt> y_topright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_topright_gt -  x_bottomleft_gt + 1) * (  y_topright_gt -y_bottomleft_gt + 1)
    Pred_bbox_area =(x_topright_p - x_bottomleft_p + 1 ) * ( y_topright_p -y_bottomleft_p + 1)
    
    x_bottom_left =np.max([x_bottomleft_gt, x_bottomleft_p])
    y_bottom_left = np.max([y_bottomleft_gt, y_bottomleft_p])
    x_top_right = np.min([x_topright_gt, x_topright_p])
    y_top_right = np.min([y_topright_gt, y_topright_p])
    
    intersection_area = (x_top_right- x_bottom_left + 1) * (y_top_right-y_bottom_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

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

def pred_splitter(path_to_data, path_to_test_ann, score_thr): 

    
    
    with open(path_to_test_ann) as json_test:
        data = json.load(json_test)
        images_name = data['images']

    scaf_dic = dict()
    plan_dic = dict()

    
    with open(path_to_data) as json_file:
        data = json.load(json_file)
        for el in data:
            if el['category_id']==1 and float(el['score'])>score_thr:
                name = images_name[int(el['image_id'])]['file_name'].split('/')[-1]
                
                if name not in scaf_dic.keys(): 
                    add = dict()
                    box = [float(el['bbox'][0]),float(el['bbox'][1]),float(el['bbox'][0])+float(el['bbox'][2]),float(el['bbox'][1])+float(el['bbox'][3])]
                    add['boxes'] = [box] 
                    add['scores'] = [float(el['score'])]
                    scaf_dic[name] = add
                else: 
                    box = [float(el['bbox'][0]),float(el['bbox'][1]),float(el['bbox'][0])+float(el['bbox'][2]),float(el['bbox'][1])+float(el['bbox'][3])]
                    scaf_dic[name]['boxes'].append(box)
                    scaf_dic[name]['scores'].append(float(el['score']))
        
            elif el['category_id']==0 and float(el['score'])>score_thr:
                name = images_name[int(el['image_id'])]['file_name'].split('/')[-1]
                if name not in plan_dic.keys(): 
                    add = dict()
                    box = [float(el['bbox'][0]),float(el['bbox'][1]),float(el['bbox'][0])+float(el['bbox'][2]),float(el['bbox'][1])+float(el['bbox'][3])]
                    add['boxes'] = [box] 
                    add['scores'] = [float(el['score'])]
                    plan_dic[name] = add
                else: 
                    box = [float(el['bbox'][0]),float(el['bbox'][1]),float(el['bbox'][0])+float(el['bbox'][2]),float(el['bbox'][1])+float(el['bbox'][3])]
                    plan_dic[name]['boxes'].append(box)
                    plan_dic[name]['scores'].append(float(el['score']))
                    
    return (scaf_dic,plan_dic)
    
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
        fn=0
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
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


        
def TP_FP_FN(path_to_true, path_to_pred, path_test_json, iou_thr, score_thr):
    '''
    Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args: - path to the file with the annotation 
          - path to the file with the prediction 
    Return:  dict: true positives (int), false positives (int), false negatives (int)
    '''
    scaf_pred, plan_pred = pred_splitter(path_to_pred, path_test_json, score_thr)
    scaf_gt, plan_gt = gt_splitter(path_to_true)
    
   
    scaf_gt = complete_test_images(scaf_gt, scaf_gt,  False)
    scaf_gt = complete_test_images(scaf_gt, scaf_pred,  False)
    plan_gt = complete_test_images(plan_gt, plan_gt, False)
    plan_gt = complete_test_images(plan_gt, plan_pred, False)
    scaf_pred = complete_test_images(scaf_pred,scaf_pred)
    scaf_pred = complete_test_images(scaf_pred,scaf_gt)
    plan_pred = complete_test_images(plan_pred,plan_pred)
    plan_pred = complete_test_images(plan_pred,plan_gt)

    tp_scaf = 0 
    fp_scaf = 0 
    fn_scaf = 0 
    for el in scaf_gt.keys(): 
        ret = get_single_image_results(scaf_gt[el], scaf_pred[el],iou_thr)
        tp_scaf += ret['true_positive']
        fp_scaf += ret['false_positive']
        fn_scaf += ret['false_negative']

    tp_plan = 0 
    fp_plan = 0 
    fn_plan = 0
    for el in plan_gt.keys(): 
        ret = get_single_image_results(plan_gt[el], plan_pred[el],iou_thr)
        tp_plan += ret['true_positive']
        fp_plan += ret['false_positive']
        fn_plan += ret['false_negative']

    return  [{'true_positive_scaf': tp_scaf, 'false_positive_scaf': fp_scaf, 'false_negative_scaf': fn_scaf}, {'true_positive_plan': tp_plan, 'false_positive_plan': fp_plan, 'false_negative_plan': fn_plan}]

def perc_of_predict(tp_scaf, tp_plan, path_to_data): 

    num_scaf = 0 
    num_plan = 0 
    with open(path_to_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[5]=='scafoideus_titanus': 
                num_scaf +=1
            if row[5]=='planococcus_ficus_m': 
                num_plan +=1 
    
    return tp_scaf/num_scaf, tp_plan/num_plan
            
def go_noARGPARSE(path_to_test, path_to_pred, path_to_test_json, iou, score_thr):
    res = TP_FP_FN(path_to_test, path_to_pred, path_to_test_json, iou, score_thr)
    scafTPFPFN = res[0]
    planTPFPFN = res[1]
    
    try: 
        mAP_scaf = scafTPFPFN['true_positive_scaf']/(scafTPFPFN['true_positive_scaf']+scafTPFPFN['false_positive_scaf'])
    except: 
        mAP_scaf = 0 

    try: 
        mAP_plan = planTPFPFN['true_positive_plan']/(planTPFPFN['true_positive_plan']+planTPFPFN['false_positive_plan'])
    except: 
        mAP_plan = 0
    
    try: 
       rc_scaf = scafTPFPFN['true_positive_scaf']/(scafTPFPFN['true_positive_scaf']+scafTPFPFN['false_negative_scaf'])
    except: 
        mAP_scaf = 0 

    try: 
        rc_plan = planTPFPFN['true_positive_plan']/(planTPFPFN['true_positive_plan']+planTPFPFN['false_negative_plan'])
    except: 
        rc_plan = 0


    rapp_ind = perc_of_predict(scafTPFPFN['true_positive_scaf'], planTPFPFN['true_positive_plan'], path_to_test)
    dic =  {'scafoideus': {'AP50': mAP_scaf, 'IR': rapp_ind[0], 'Recall': rc_scaf}, 'planococcus_ficus_m': {'AP50': mAP_plan, 'IR': rapp_ind[1], 'Recall': rc_plan}, 'general': {'MAP': (mAP_scaf + mAP_plan)/2, 'Recall': (rc_scaf + rc_plan)/2}}

    return dic
def main(): 
    parser = argparse.ArgumentParser(description='Optimiaze the confidence threshold')
    parser.add_argument('--path_to_test', type = str , help = 'path to test .csv annotation')
    parser.add_argument('--path_to_pred', type = str , help = 'path to predictions')
    parser.add_argument('--path_to_test_json', type = str , help = 'path to test .json annotation')
    parser.add_argument('--iou', type=float , help = 'IoU theshold', default=0.5)
    parser.add_argument('--steps', type=float , help = 'increasing steps of score theshold that we want to evaluate', default=0.5)
    args = parser.parse_args()


    score_thr = np.arange(0.0, 1.0, args.steps)
    print(score_thr)
    AP_scaf = []
    AP_plan = []
    rc_scaf = []
    rc_plan = []
    IR_scaf = []
    IR_plan = []
    mAP = []
    RC = []

    for st in score_thr: 
        out = go_noARGPARSE(args.path_to_test, args.path_to_pred, args.path_to_test_json, args.iou, st)
        AP_scaf.append(out['scafoideus']['AP50'])
        rc_scaf.append(out['scafoideus']['Recall'])
        AP_plan.append(out['planococcus_ficus_m']['AP50'])
        rc_plan.append(out['planococcus_ficus_m']['Recall'])
        IR_scaf.append(out['scafoideus']['IR'])
        IR_plan.append(out['planococcus_ficus_m']['IR'])
        mAP.append(out['general']['MAP'])
        RC.append(out['general']['Recall'])

    mAP = np.array(mAP)
    RC = np.array(RC)
    # We maximize the F-Measure
    ind_max = np.argmax(2*mAP*RC/(mAP+RC)) 
   

    print('------------------------ Best Model using ROC------------------------ ')
    print('')
    print('The best model according to ROC is the one with threshold ' + str(score_thr[ind_max]) + '. Below the statistics are vailable')
    print('')
    print('------------- SCAFOIDEUS TITANIUS ------------- ')
    print('')
    print('AP' + str(args.iou)+':     ' + str(AP_scaf[ind_max]))
    print('Recall' + str(args.iou)+':     ' + str(rc_scaf[ind_max]))
    print('IR:     '+ str(IR_scaf[ind_max]) )
    print('')
    print('------------- PLANOCOCCUS_FICUS_M ------------- ')
    print('')
    print('AP' + str(args.iou)+':     ' + str(AP_plan[ind_max]))
    print('Recall' + str(args.iou)+':     ' + str(rc_plan[ind_max]))
    print('IR:     '+ str(IR_plan[ind_max]) )
    print('')
    print('------------- GENERAL ------------- ')
    print('')
    print('mAP'+ str(args.iou)+':     ' + str(mAP[ind_max]))
    print('Recall'+ str(args.iou)+':     ' + str(RC[ind_max]))

main()

