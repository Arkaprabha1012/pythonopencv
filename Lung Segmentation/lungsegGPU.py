import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt
import cv2
from keras.preprocessing.image import save_img
import pylab as p
import matplotlib.cm as cm

def loadDataGeneral(df, path, im_shape):
    X = []
    filen=[]
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[1]))
        filen.append(item[1])
        print(item[1]," loaded")
        if len(img.shape)==3 and img.shape[2]==3:
          #print(img.shape)
          img = np.float32(img)
          img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        X.append(img)
        if i==15474:
          break
    print("done")    
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print('### Dataset loaded')
    print('\t{}'.format(path))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    #print(len(filen))
    #print(filen[0])
    return X,filen
if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path ='alldata.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path ='Allimages/all_images/'

    df = pd.read_csv(csv_path)
    df.head()
    
    # Load test data
    im_shape = (256, 256)
    X,filen1 = loadDataGeneral(df, path, im_shape)
    #print(filen1[0])
    #n_test = X.shape[0]
    inp_shape = X[0].shape
    # Load model
    model_name = 'trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)
    i = 0
    #plt.figure(figsize=(10, 10))
    for xx,filen in test_gen.flow(X,filen1, batch_size=1):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        norm_image = cv2.normalize(pred, None, alpha = 0.0, beta = 255.0, norm_type = cv2.NORM_MINMAX)
        pred = norm_image.astype(np.uint8)
        ret,thresh2 = cv2.threshold(pred,127,255,cv2.THRESH_BINARY)
        result2 = (img * thresh2)
        plt.imshow(result2,cmap="gray")
        fig=plt.gcf()
        i+=1
        filename='Allimages/Segmented/'+filen1[i]
        print(filen1[i]," Segmentation Done.")
        plt.savefig(filename,dpi=100)
        print(filen1[i]," Segmentation Done.")
        if i == 15474:
            break
