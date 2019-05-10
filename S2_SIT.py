import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# SIT model is defined here, applied to masked raster

def colorbar(mappable, title):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(mappable, cax=cax)
    clb.ax.set_title(title)
    return clb
def S2_SIT_model(S2_ndarray, cloud_mask, land_mask, BOA=False ,display=True):
    """

    :param S2_ndarray: Image from raster
    :param masks: list of masks with same image dimensions as input S2_ndarray
    :param BOA: Different models for BOA and TOA reflectances
    :return: SIT_mask
    """
    if BOA:
        SIT_model_fp = 'models/BOA_LGBM_Model.txt'
    else:
        SIT_model_fp = 'models/TOA_LGBM_Model.txt'
        valid_bands =[0,1,2,3,4,5,6,7,8,9,11,12]
        S2_ndarray = S2_ndarray[..., valid_bands]
    rgb = S2_ndarray[..., [3, 2, 1]]
    data_flat = S2_ndarray.reshape(S2_ndarray.shape[0] * S2_ndarray.shape[1], -1)
    cloud_mask_flat = cloud_mask.reshape(S2_ndarray.shape[0] * S2_ndarray.shape[1])
    land_mask_flat = land_mask.reshape(S2_ndarray.shape[0] * S2_ndarray.shape[1])
    data_flat_no = data_flat[(data_flat[..., 0] != 0) & (~cloud_mask_flat) & (~land_mask_flat), ...]

    model = lgb.Booster(model_file=SIT_model_fp)

    SIT_estimation = model.predict(data_flat_no)
    SIT_img_flat = np.zeros((S2_ndarray.shape[0] * S2_ndarray.shape[1]), dtype='float32')
    SIT_img_flat[(data_flat[..., 0] != 0) & (~cloud_mask_flat) & (~land_mask_flat)] = SIT_estimation
    SIT_img = SIT_img_flat.reshape(S2_ndarray.shape[0], S2_ndarray.shape[1])
    SIT_img[SIT_img < 0.0] = 0
    SIT_img_dn= SIT_img
    SIT_img_dn[cloud_mask] = 40
    SIT_img_dn[land_mask] = 50
    SIT_img_dn = (SIT_img_dn*1000).astype('uint16')

    if display:
        cmap = plt.get_cmap('inferno')
        cmap.set_bad('black')
        SIT_masked = np.ma.array(SIT_img, mask= (cloud_mask) & (land_mask))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
        im1 = ax1.imshow(SIT_masked, cmap=cmap,vmin=0, vmax=30)
        colorbar(im1, "SIT (cm)")
        im2 = ax2.imshow(rgb)
        plt.tight_layout()
        ax1.set_title("Thin SIT Prediction from LGBM model")
        ax2.set_title("RGB composite")
        plt.show()

    return SIT_img_dn
