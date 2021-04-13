from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import csv




def run_prediction(model,img_path, file_pred): 
    ''' Args: - The model used for the prediction 
              - The path of the image on which we want to do the prediction 
              - name of the file to saving the prediction 
        This function write on a .csv file the predicted bounded box.
    '''
    result = inference_detector(model, img_path)
    scaf_pred = result[0]
    plan_pred = result[1]

    with open(file_pred, mode='a') as csvfile:
        csvfile_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for scaf in scaf_pred:
            csvfile_writer.writerow([img_path,scaf[0],scaf[1],scaf[2],scaf[3],'scafoideus_titanus',scaf[4]])

        for plan in plan_pred:
            csvfile_writer.writerow([img_path,plan[0],plan[1],plan[2],plan[3],'planococcus_ficus_m',plan[4]])
            
    