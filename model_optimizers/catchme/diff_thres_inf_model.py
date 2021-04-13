# This function builds different models according to differents values of the NMS threshold
import os

def diff_thres_inf_model(path_to_train, path_to_save, thresholds): 
    ''' Args: - path to the trained model (weights)
              - path to save the inference models 
              - values for the threshold 
        Return: save in path_to_save the different inference models
    '''
    if not isinstance(thresholds,list): 
        thresholds = list(thresholds)
    
    for t in thresholds:
        name = '/catchme_12_2020_inf_'+str(t)+'.h5'
        command = 'retinanet-convert-model ' + path_to_train + ' ' + path_to_save + name + ' ' + '--nmsThreshold=' + str(t)
        os.system(command)
