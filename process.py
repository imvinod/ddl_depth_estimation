import glob
import numpy as np
from PIL import Image
from src.tests import *
import time
from sklearn.decomposition import PCA


def RMSE(target, prediction):
    return np.sqrt(np.mean((target - prediction)**2))

def load_data_from_folder(folder, isDepth = False):
    filelist = glob.glob(folder+'/*.png')
    data = []
    for fname in filelist:
        png = np.array(Image.open(fname), dtype=int)
        if isDepth:        
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert(np.max(png) > 255)
            depth = png.astype(np.float) / 256.
            depth[png == 0] = -1.
            data.append(depth)
        else:
            data.append(png)
    return np.asarray(data)

def measure_duration(X):
    preds = []
    durations = []
    for img in X:
        start = time.time()
        depth = run(img)
        end = time.time()
        durations.append(end - start)
        preds.append(depth)
    return (sum(durations[1:]) / (len(durations)-1)), preds

def getDepth(img):
    start = time.time()
    depth = run(img)
    end = time.time()
    return end-start, depth

def pca(X):
    pca_imgs = []
    for i, img in enumerate(X): 
        pca_img = img.reshape(375, -1 )
        # Increasing n components will increance explained variance but will decrease our accuracy benefits.
        pca = PCA(n_components = 64, svd_solver='randomized').fit(pca_img)
        pca_img = pca.transform(pca_img)
#         print(pca_img.shape )
#         print("Retained variance", np.sum(pca.explained_variance_ratio_))
        img = pca.inverse_transform(pca_img)
        img = img.reshape(375, 1242, 3)
        pca_imgs.append(img)
    return pca_imgs


def frameFiltering(X, enabled =True):
    all_preds = []
    durations = []

    for i, img in enumerate(X):   
        if enabled:
            if i%5 == 0:  # Every 5 images recreate the mask to ignore the pixels that are too far away to be considered
                duration, pred = getDepth(img)           
                d_mask = pred > 15    # create a mask where all pixels value above 15 is true for the top 100 pixes, True elsewhere
                d_mask[100:] = True
            else:
                duration, pred = getDepth(img*d_mask[:, :, None])
        else:
            duration, pred = getDepth(img)
        all_preds.append(pred)
        durations.append(duration)
        
    avg_duration = (sum(durations[1:]) / (len(durations)-1))  #ignoring the duration for first image, it takes abnormally high time
    return avg_duration, all_preds


def main():
    val_folder = "examples/kitti/val/"
    depths_folder = val_folder + "depths"
    images_folder = val_folder + "images"
    epsilon = 0.000001

    # Load images
    X = load_data_from_folder(images_folder)
    Y = load_data_from_folder(depths_folder, True)

    #print(X.shape)
    avg_duration_with_filtering, preds_with_filtering = frameFiltering(X, True) 
    avg_duration_without_filtering, preds_without_filtering = frameFiltering(X, False)    
    improvement = round((avg_duration_without_filtering - avg_duration_with_filtering)*100 / (avg_duration_without_filtering + epsilon), 3)
    print("Percentage improvement in execution time with Frame Filtering: ", improvement)       
    
    pca_imgs = pca(X)
    avg_duration_with_pca, preds = measure_duration(pca_imgs)
    avg_duration_without_pca, preds = measure_duration(X)
    improvement = round((avg_duration_without_pca - avg_duration_with_pca)*100 / (avg_duration_without_pca +epsilon), 3)
    print("Percentage improvement in execution time with PCA: ", improvement)    
        
    
if __name__ == '__main__':
    main()
    