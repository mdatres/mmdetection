import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import csv

def plot_result(result, img_path, test_set, scr_thr):
    
    img = np.array(Image.open(img_path), dtype=np.uint8)
    gt_boxes = [] 
    with open(test_set, newline='') as f:
        reader = csv.reader(f)
        for row in reader: 
            if row[0] == img_path: 
                add = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), row[5]]
                gt_boxes.append(add)
    
   
    fig, ax = plt.subplots(figsize=(40, 30))
    plt.imshow(img)
    ax.imshow(img)
    for el in gt_boxes: 
        rect = patches.Rectangle((el[0], el[1]), el[2]-el[0], el[3]-el[1], linewidth=3, edgecolor='green',facecolor="none")
        ax.text(el[0], el[1], el[4], fontsize=7)
        ax.add_patch(rect)
    
    scaf_pred = result[0]
    plan_pred = result[1]
    for pr in scaf_pred:
        if pr[4] > scr_thr:
            rect = patches.Rectangle((pr[0], pr[1]), pr[2]-pr[0], pr[3]-pr[1], linewidth=2, edgecolor='red',facecolor="none")
            ax.text(pr[0], pr[1], 'scafoideus_titanus, ' + str(pr[4]), fontsize=7)
            ax.add_patch(rect)
    for pr in plan_pred:
        if pr[4] > scr_thr:
            rect = patches.Rectangle((pr[0], pr[1]), pr[2]-pr[0], pr[3]-pr[1], linewidth=2, edgecolor='red',facecolor="none")
            ax.text(pr[0], pr[1], 'planococcus_ficus_m, ' + str(pr[4]), fontsize=7)
            ax.add_patch(rect)

    plt.show()





