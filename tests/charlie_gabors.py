import os
import numpy as np
import scipy as sp
from tqdm import tqdm


rot = np.array([90,-67.5,-45,-22.5,0,22.5,45,67.5])
RF_siz = [15]
Div = sp.arange(4,3.15,-.05)


def create_gabor(rot,RF_siz,Div,plot,num=10):
    count = 0
    numFilterSizes   = len(RF_siz)
    numSimpleFilters = len(rot)
    lamb = (RF_siz[0] * 2)/4.0
    sigma  = [lamb * 0.8]
    G      = 0.3
    phases = [0]

    # Initialize Filterbank
    alt_fb = np.zeros((RF_siz[0],RF_siz[0],1,len(rot)),dtype=np.float32)


    # for k in tqdm(range(0,numFilterSizes-1)):
    for k in tqdm(range(1,numFilterSizes+1)):
        for r in tqdm(range(1,numSimpleFilters+1)):

            f = np.zeros([RF_siz[numFilterSizes-1],RF_siz[numFilterSizes-1]])
            fx = np.zeros([RF_siz[numFilterSizes-1],RF_siz[numFilterSizes-1]])

            ## Parameters
            theta     = rot[r-1]*(np.pi/180)
            filtSize  = RF_siz[k-1]

            img_center = np.ceil(filtSize/2.0) ## New center for padding with zeros

            center    = np.ceil(filtSize/2.0) ## Old and possibly more accurate center


            filtSizeL = center-1
            filtSizeR = filtSize-filtSizeL-1
            sigmaq    = (sigma[k-1]) * (sigma[k-1])


            # Compute filter values

            for i in range(int(-1 * filtSizeL),int(filtSizeR+1)):
                for j in range(int(-1 * filtSizeL),int(filtSizeR+1)):

                    if (np.sqrt((i**2)+(j**2))>(filtSize/2 )) :
                        E = 0
                    else :
                        x = i*np.cos(theta) - j*np.sin(theta)
                        y = i*np.sin(theta) + j*np.cos(theta)

                        E = np.exp((-1*((x**2)+ (G**2) * (y**2)))/(2*sigmaq))*np.cos(2*np.pi*x/lamb + phases)

                    f[int(j+img_center-1),int(i+img_center-1)] = E


            ## Append to fb (filterbank)
            f = f - np.mean(np.mean(f))
            f = f / np.sqrt(np.sum(np.sum(f**2)))

            # Reshaped image
            alt_fb[:,:,0,count] = f
            count += 1

            if (plot):
                if count % num == 0:
                    plt.imshow(f,cmap='Greys')
                    plt.show()

    return (np.array(alt_fb))



## Create Dictionary ---------------

gabor_array = create_gabor(rot,RF_siz,Div,False)

bias = []

gabor_dictionary = {}
gabor_dictionary['s1'] = gabor_array,bias

f,x = gabor_dictionary['s1']

# for idx in range(f.shape[-1]):
#    plt.subplot(1, f.shape[-1], idx + 1)
#      plt.imshow(f.squeeze()[:, :, idx])
# plt.show()

## Save Dictionary ----------------------
out_name = 'gabors_for_contours_15.npy'
out_path = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'clicktionary',
    'pretrained_weights',
    out_name)
np.save(out_path, gabor_dictionary)


