import csv
import json
import numpy as np
import argparse
import os 
from run_prediction import run_prediction
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from TP_FP_FN import gt_splitter, pred_splitter, complete_test_images, get_single_image_results
from metrics import countersPercentage

def TP_FP_FN_diff(path_to_true, path_to_pred, iou_thr):
    '''
    Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args: - path to the file with the annotation 
          - path to the file with the prediction 
    Return:  dict: true positives (int), false positives (int), false negatives (int)
    '''
    scaf_pred, plan_pred = pred_splitter(path_to_pred)
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
    return  {'true_positive_scaf': tp_scaf, 'false_positive_scaf': fp_scaf, 'false_negative_scaf': fn_scaf,'true_positive_plan': tp_plan, 'false_positive_plan': fp_plan, 'false_negative_plan': fn_plan}

def test_model():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--path-to-test', type = str , help = 'path to test .csv annotation')
    parser.add_argument('--checkpoint', type = str , help = 'path to the .pth fike')
    parser.add_argument('--cfg', type = str , help = 'path to configuration file')
    parser.add_argument('--device', type = str, help = 'device to be used', default='cuda:0')
    parser.add_argument('--iou', type=float , help = 'IoU theshold', default=0.5)
    
    args = parser.parse_args()


    config_file = args.cfg
    checkpoint_file = args.checkpoint
    model = init_detector(config_file, checkpoint_file, device=args.device)

    print('The model has been loaded succesfully')
    
    test_images = []
    with open(args.path_to_test) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                n = row[0]
                test_images.append(n)
    
    test_images = list(set(test_images))

    try: 
        os.remove("test_pred")
    except: 
        pass

    print('Start predicting')
    
    for im in test_images:
        print(im)
        run_prediction(model, im, "test_pred")

    
    dic = TP_FP_FN_diff(args.path_to_test, "test_pred", args.iou)
    maP_scaf = dic['true_positive_scaf']/(dic['true_positive_scaf']+dic['false_positive_scaf'])
    maP_plan = dic['true_positive_plan']/(dic['true_positive_plan']+dic['false_positive_plan'])
    maP = (maP_plan+maP_scaf)/2

    CSI_scaf = dic['true_positive_scaf']/(dic['true_positive_scaf']+dic['false_positive_scaf'] + dic['false_negative_scaf'])
    CSI_plan = dic['true_positive_plan']/(dic['true_positive_plan']+dic['false_positive_plan']+ dic['false_negative_scaf'])
    CSI = (CSI_plan+CSI_scaf)/2

    FAR_scaf = dic['false_positive_scaf']/(dic['true_positive_scaf']+dic['false_positive_scaf'])
    FAR_plan = dic['false_positive_plan']/(dic['true_positive_plan']+dic['false_positive_plan'])
    FAR = (FAR_plan+FAR_scaf)/2

    REC_scaf = dic['true_positive_scaf']/(dic['true_positive_scaf']+dic['false_negative_scaf'])
    REC_plan = dic['true_positive_plan']/(dic['true_positive_plan']+dic['false_negative_plan'])
    REC = (REC_plan+REC_scaf)/2

    counterDic = countersPercentage(args.path_to_test, "test_pred")
    counterScaf = counterDic['scafoideus_titanus']
    counterPlan = counterDic['planococcus_ficus_m']




    print('---------------------------- Model Metrics -------------------------------')
    print('')
    print('mAP_scafoideus:  ' + str(maP_scaf))
    print('mAP_plan:  ' + str(maP_plan))
    print('general mAP  ' + str(maP))
    print('')
    print('CSI_scaf:  ' + str(CSI_scaf))
    print('CSI_plan:  ' + str(CSI_plan))
    print('general CSI:  ' + str(CSI))
    print('')
    print('FAR_scaf:  ' + str(FAR_scaf))
    print('FAR_plan:  ' + str(FAR_plan))
    print('general FAR:  ' + str(FAR))
    print('')
    print('Recall_scaf:  ' + str(REC_scaf))
    print('Recall_plan:  ' + str(REC_plan))
    print('general Recall:  ' + str(REC))
    print('')
    print('Percentage_scaf:  ' + str(counterScaf))
    print('Percentage_plan:  ' + str(counterPlan))

test_model()

    
    
    