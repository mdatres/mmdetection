import numpy as np

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