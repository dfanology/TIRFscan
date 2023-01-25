"""

Astigmatic Gaussian PSF is generated using the napari PSF generator plugin
https://www.napari-hub.org/plugins/napari-psf-simulator

Using settings:
NA=1.33
n=1.33
wavelength: 0.68 um (typical emission for 640nm excitation)
Nxy: 127
Nz: 51
dxy: 0.05 um
dz: 0.02 um

With Zernike abberation:
N=2, M=2
weight: 0.5 lambda,

then saving data using napari python console and:

np.save('astig_psf_17NA.npy', viewers.layers[0].data)

"""
import os
from photonpy import Context
import numpy as np
import matplotlib.pyplot as plt
from photonpy.cpp.gaussian import GaussianPSFMethods, Gauss3D_Calibration

import tifffile
from scipy.optimize import least_squares

def make_summed(fn, imgs_per_plane=20):
    tif = tifffile.imread(fn)
    return tif.reshape((-1,imgs_per_plane,6,tif.shape[-2],tif.shape[-1]))


def calibrate_gaussian_3D(rois, zpos):    
    roisize = rois.shape[-1]
    with Context() as ctx:
        gpm = GaussianPSFMethods(ctx)
        psf = gpm.CreatePSF_XYIBgSigmaXY(roisize, 2, cuda=False)
        
        params = psf.Estimate(rois)[0]
        
        plt.figure()
        plt.plot(params[:,4:])
    
        def f(p, z):
            s0, gamma, d, A = p
            return s0 * np.sqrt(1 + (z - gamma)**2 / d**2 + A * (z - gamma)**3/d**2)
    
        def func(p, z, y):
            return f(p, z) - y
        
        sigma_xs = params[:,4]
        sigma_ys = params[:,5]
        
        bounds = ((1, -600, 0, 1e-5), (10, 600, 1e3, 1e-1))
        p0 = [2, 0, 3e2, 1e-5]
        p_x = least_squares(func, p0, loss='huber', bounds=bounds, args=(zpos, sigma_xs))
        p_y = least_squares(func, p0, loss='huber', bounds=bounds, args=(zpos, sigma_ys))
        
        plt.figure()
        plt.plot(zpos, sigma_xs, 'x', label='$\sigma_x$')
        plt.plot(zpos, sigma_ys, 'x', label='$\sigma_y$')
        plt.plot(zpos, f(p_x.x, zpos), '--', label='x fit')
        plt.plot(zpos, f(p_y.x, zpos), '--', label='y fit')
        #plt.ylim([0, 10])
        plt.legend()
        plt.savefig('psf-fit.svg')
    
        calib = Gauss3D_Calibration(p_x.x, p_y.x)
    
        p = np.zeros((len(zpos), 5))
        p[:,[0,1,3,4]] = params[:,:4]
        p[:,2] = zpos
    
        psf_astig = gpm.CreatePSF_XYZIBg(rois.shape[-1], calib, cuda=False)
        ev = psf_astig.ExpectedValue(p)
        
        #image_view(ev)
        return calib, ev


fn = 'psf_astig05lam_dz002_dx01_lam068_NA133.npy'
zstep = 0.02
stack = np.load(os.path.split(__file__)[0]+"/"+ fn)
W=stack.shape[-1]
L = len(stack)

S=12
D = 10

stack = stack[L//2-D:L//2+D+1,W//2-S:W//2+S, W//2-S:W//2+S]

#%%
with Context(debugMode=False) as ctx:
    gaussian = GaussianPSFMethods(ctx)

    numsteps = stack.shape[0]
    zpos = np.linspace(-zstep * numsteps/2, zstep * numsteps/2, numsteps)
    calib,ev = calibrate_gaussian_3D(stack, zpos)
    calib.save('gausscalib.npy')
    
    d = calib.__dict__()
    print(f'calibration: {d}')

    Z = len(stack)//2
    fig,ax=plt.subplots(1,2)
    ax[0].imshow(stack[Z])
    ax[0].set_title('Simulation')

    ax[1].imshow(ev[Z])
    ax[1].set_title('Fit')

    plt.savefig('psf-fit-rois.svg')
    