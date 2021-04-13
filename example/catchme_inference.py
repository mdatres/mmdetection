from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '/home/datres/Catchme/mmdetection/configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
checkpoint_file = '/data0/work_dirs/VFNet/epoch_40.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/data0/Catchme/dataset/16_03_2021/images/20201031_catchme-iori_g_-_sx_2_.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
