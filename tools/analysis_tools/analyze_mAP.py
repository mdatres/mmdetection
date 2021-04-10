import argparse
import matplotlib.pyplot as plt
import numpy as np

def argmax(lst):
  return lst.index(max(lst))


def plot_mAP():
    parser = argparse.ArgumentParser(description='Print the best model')
    parser.add_argument('--logs', nargs="+" , help = 'path to .csv to convert')
    args = parser.parse_args()


    
    mAP50 = list()
    initial = 0 
    i = 0 

    with open(args.logs[0], 'r') as log:
        lines = log.readlines()
        for l in lines: 
            if 'Epoch(val)' in l: 
                if i==0: 
                    initial = int(l.split(' [')[1][0:2])
                    i +=1
                try:
                    add = l.split('bbox_mAP_50:')[1].split(',')[0]
                    print(l.split('bbox_mAP_50:')[1].split(',')[0])
                    mAP50.append(add)
                except: 
                    pass
    
    
    mAP50 = [float(r) for r in mAP50]
    print('--------------------------------------------------------------------------------------------')
    print('The best model according to the bbox mAP with IoU 0.5 is at epoch ' + str(initial + argmax(mAP50)) + ' with mAP ' + str(mAP50[argmax(mAP50)]))

    
    xi = list(range(len(mAP50)))
    plt.plot(xi, mAP50, 'bo', xi, mAP50, 'k')
    plt.show()



plot_mAP()