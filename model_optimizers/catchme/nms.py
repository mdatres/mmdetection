# Define the non-maximum-supression algorithm (NMS)
import numpy as np
#from IoU import calc_iou
import csv

def create_dic_pred(path): 
    d = dict()
    with open(path, mode='r') as true:
        reader = csv.reader(true)
        for r in reader: 
            name = r[0]
            if name not in d.keys(): 
                add = dict()
                box = [float(r[1]),float(r[2]),float(r[3]),float(r[4]),r[5]]
                add['boxes'] = [box] 
                add['scores'] = [float(r[6])]
                d[name] = add
            else: 
                box = [float(r[1]),float(r[2]),float(r[3]),float(r[4]),r[5]]
                d[name]['boxes'].append(box)
                d[name]['scores'].append(float(r[6]))
    return d

def nms(pred_boxes_path, ov_thres):
    ''' Args: - the predicted boxes
              - the overlap threshold
              - scores of the predicted bbox
        Return: - a list of the filtered bounded boxes
    '''
    # Read the boxes 
    dic = create_dic_pred(pred_boxes_path)

    for el in dic.keys():
        pred_boxes = dic[el]['boxes']
        pred_scores = dic[el]['scores']
        
        scores = np.array(pred_scores)
        
        # if there are no boxes, return an empty list
        if len(pred_boxes) == 0:
            return []
        # initialize the list of picked indexes
        D = []
        x1 = np.array([float(pb[0]) for pb in pred_boxes])
        y1 = np.array([float(pb[1]) for pb in pred_boxes])
        x2 = np.array([float(pb[2]) for pb in pred_boxes])
        y2 = np.array([float(pb[3]) for pb in pred_boxes])
        
        # sort the vector score (or equivalentely the bboxes)
        idxs = np.argsort(-scores)
        #idxs = np.array(idxs.tolist(), dtype = np.int32)

        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[idxs[1:]] - inter)

            # Return the idxs that corresponds to a bounding box with lower threshold (the ones that we want to keep) 
            inds = np.where(ovr <= ov_thres)[0]

            idxs = idxs[inds + 1]

        D = [(pred_boxes[k],pred_scores[k]) for k in keep]
        
    
        with open('nms_pred', mode='a') as csvfile:
            csvfile_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for d in D:
                add = [el]
                add.extend(d[0])
                add.extend([d[1]])
                csvfile_writer.writerow(add)
    
