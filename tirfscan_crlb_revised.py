# -*- coding: utf-8 -*-
"""
CRLB calculation for TIRFscan
@authors: Jelmer, Daniel, Carlas

Revised 2023-01-25:
    Added focal plane (219-231)
    Added different PSF models (22-57, 193)
    Added plot of ROI for debugging and visualization (245-255)
    Added manuscript figure generation
"""
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt

####################
# Sets up the imaging model
#----------------------------------------------------

# Calibration data for astigmatic imaging
#----------------------------------------
def gauss3D_calib(NA):
    
    # For simulated PSF with NA=1.33
    if(NA == 133):
        return [[1.0, 
                 0.19020476937294006, 
                 0.15444783866405487, 
                 0.10000000149011612],
                [1.0,
                 -0.1928471326828003, 
                 0.15837538242340088, 
                 9.999999747378752e-06]]


    # For simulated PSF with NA=1.7
    elif(NA == 17): 
        return [[1.0, 
                 0.13603557646274567, 
                 0.18010391294956207, 
                 9.999999747378752e-06],
                [1.0, 
                 -0.13707047700881958, 
                 0.1833316683769226, 
                 0.10000000149011612]]


    # For simulated PSF with NA=1.5
    elif(NA == 15): 
        return [[1.0, 
                 0.1645256131887436, 
                 0.17072711884975433, 
                 9.999999747378752e-06],
                [1.0, 
                 -0.16463486850261688, 
                 0.17093676328659058, 
                 0.005561515688896179]]

# Calculation of mu and first derivative for astigmatic imaging
#--------------------------------------------------------------    
def compute_derivatives_XYZIBg(theta, numpixels, calib):
    
    # Initialisation of variables
    tx = theta[:,0][:,None,None]
    ty = theta[:,1][:,None,None]
    tz = theta[:,2][:,None,None]
    tI = theta[:,3][:,None,None]
    tbg = theta[:,4][:,None,None]
    
    # Astigmatic imaging variables
    s0_x, g_x, d_x, A_x = calib[0]
    s0_y, g_y, d_y, A_y = calib[1]
    tzx = tz - g_x
    tzy = tz - g_y
    sg_x = np.sqrt(1 + tzx**2 / d_x**2 + A_x * tzx**3 / d_x**2)
    sg_y = np.sqrt(1 + tzy**2 / d_y**2 + A_y * tzy**3 / d_y**2)
    
    # Prefactors for easy calculations
    One2PiSigma_x = 1 / (np.sqrt(2 * np.pi) * sg_x * s0_x)
    One2Sigma_x = 1 / (np.sqrt(2) * sg_x * s0_x)
    One2PiSigma_y = 1 / (np.sqrt(2 * np.pi) * sg_y * s0_y)
    One2Sigma_y = 1 / (np.sqrt(2) * sg_y * s0_y)
        
    # 1. IMAGING MODEL
    #-----------------
    # Pixel centers
    # Necessary for Fisher matrix
    pixelpos = np.arange(0, numpixels)
    Xc, Yc = np.meshgrid(pixelpos, pixelpos)
    
    # Change of variables that are used often
    Xp0 = (Xc[None]-tx+0.5)
    Xm0 = (Xc[None]-tx-0.5) 
    Yp0 = (Yc[None]-ty+0.5)
    Ym0 = (Yc[None]-ty-0.5)
    Xp1 = Xp0 * One2Sigma_x
    Xm1 = Xm0 * One2Sigma_x
    Yp1 = Yp0 * One2Sigma_y
    Ym1 = Ym0 * One2Sigma_y
    
    # E and mu
    Ex = 0.5 * sps.erf(Xp1) - 0.5 * sps.erf(Xm1)
    Ey = 0.5 * sps.erf(Yp1) - 0.5 * sps.erf(Ym1)
    mu = tbg + tI * Ex * Ey
    
    # 2. FIRST DERIVATIVES
    #---------------------
    # dx, dy, dI, and dbg since they are easier
    dEx = One2PiSigma_x * (np.exp(-Xm1**2) - np.exp(-Xp1**2))
    dEy = One2PiSigma_y * (np.exp(-Ym1**2) - np.exp(-Yp1**2))
    dmu_dx = tI * dEx * Ey
    dmu_dy = tI * dEy * Ex
    dmu_dI = Ex * Ey
    dmu_dbg = Ex*0 + 1

    # dz: sigma vs. theta
    dsgx_dtz = s0_x * (2*tzx/d_x**2 + 3*A_x*tzx**2/d_x**2) / (2*sg_x)
    dsgy_dtz = s0_y * (2*tzy/d_y**2 + 3*A_y*tzy**2/d_y**2) / (2*sg_y)
    
    # dz: mu vs. sigma
    G21x = One2PiSigma_x*(Xm0*np.exp(-Xm1**2)-Xp0*np.exp(-Xp1**2))/(sg_x*s0_x)
    G21y = One2PiSigma_y*(Ym0*np.exp(-Ym1**2)-Yp0*np.exp(-Yp1**2))/(sg_y*s0_y)
    dmu_dsgx = tI * Ey * G21x
    dmu_dsgy = tI * Ex * G21y
    
    # dz: mu vs. theta    
    dmu_dz = dmu_dsgx * dsgx_dtz + dmu_dsgy * dsgy_dtz

    return mu, np.array( [ dmu_dx, dmu_dy, dmu_dz, dmu_dI, dmu_dbg ] )

# Excitation field in z-direction
# returns excitations ordered by (thetas, tirf angles)
# Note: normally it is written exp(-z/d), therefore in this case kz = -1/2d
#--------------------------------------------------------------------------
def excitation_z(theta, kz):
    return np.exp(2*kz[None]*theta[:,2,None])

# Calculation of TIRFscan mu and jacobian
#----------------------------------------
def compute_derivatives_ts_XYZIBg(theta, numpixels, calib, kz, alpha):
    
    # Initialisation of variables used
    I = theta[:,3][:,None,None,None]
    bg = theta[:,4][:,None,None,None]
    
    # Excitation field
    ex = alpha * excitation_z(theta, kz) + 1 - alpha #alpha here
    dex_dz = 2*kz*alpha*excitation_z(theta, kz) 
    ex = ex[:,:,None,None]
    dex_dz = dex_dz[:,:,None,None]
    
    # Theta with intensity and bg set to 1,0 
    theta_psf = theta*1
    theta_psf[:,3] = 1
    theta_psf[:,4] = 0
    
    # Returns the normalised mu and jacobian of theta
    psf, (dpsf_dx, dpsf_dy, dpsf_dz, dpsf_dI, dpsf_dbg) = (
        compute_derivatives_XYZIBg(theta_psf,numpixels,calib))
    
    # Different backgrounds for each TIRF angle (scaled value)
    # Note: this can also be 1/kz*np.sum(1/kz)
    bg_per_angle = (kz[0]/kz)/np.sum(kz[0]/kz)
    bg_per_angle = bg_per_angle[None,:,None,None]
    
    # Final imaging model
    mu = I * (ex * psf[:,None]) + (bg * bg_per_angle)
    
    # Calculate each individual jacobian term    
    jac = np.zeros((5, len(theta), len(kz), numpixels, numpixels))
    jac[0] = I * ex * dpsf_dx[:,None]
    jac[1] = I * ex * dpsf_dy[:,None]
    jac[2] = I * (ex * dpsf_dz[:,None] + dex_dz * psf[:,None])
    jac[3] = ex * psf[:,None]
    jac[4] = bg_per_angle
    return mu, jac

####################
# Computes the CRLB
#----------------------------------------------------
def compute_crlb(mu, jac):
    mu[mu<1e-9] = 1e-9
    K = len(jac)
    sampledims = tuple(np.arange(1,len(mu.shape)))
    fi = np.zeros((len(mu),K,K))
    for i in range(K):
        for j in range(K):
            fi[:,i,j] = np.sum( 1/mu * (jac[i] * jac[j]), axis = sampledims)        
    return np.sqrt(np.linalg.inv(fi).diagonal(axis1=1,axis2=2))

# CRLB calculation
#-------------------------------------------------------------------
def crlb_study(zmax, NARange, ncovglass, alpha, fp=0, psf=15):
    
    # Generally these never change
    roisize = 16
    nwater = 1.33 
    ill_lambda = .640
    I = 1000
    bg = 10
    
    k0=2*np.pi/ill_lambda
    zrange = np.linspace(0,zmax,int(zmax*1000))
    npos = len(zrange)
    TIRFanglerange = np.arcsin(NARange/ncovglass)
    kz=-1.*k0*np.sqrt((ncovglass*np.sin(TIRFanglerange))**2-nwater**2)   
    
    # Initialize 
    crlb_astig = np.zeros((npos,5))
    crlb_ts = np.zeros((npos,5))
    crlb_if = np.zeros((npos,1))
    exc = np.zeros((npos,len(kz)))         
    
    # For each z-position, calculate CRLB
    for i in range(npos):
        
        # Position of point = theta = in the middle
        th = np.array([[roisize/2,roisize/2,zrange[i],I,bg]])
        thf = th
        thf[:,2] = th[:,2]-fp # Theta with shifted focal plane

        # Calculate CRLB for just astigmatic imaging
        expval,jac=compute_derivatives_XYZIBg(thf,
                                              roisize,
                                              gauss3D_calib(psf))
        crlb_astig[i] = compute_crlb(expval,jac)[0]*1000
        
        # Calculate CRLB for tirfscan with astigmatic imaging
        exc[i,:] = alpha*excitation_z(th, kz) + 1 -alpha #alpha here
        thf[:,3] = I/ np.sum(exc[i])
        expval_ts,jac_ts=compute_derivatives_ts_XYZIBg(thf,
                                                       roisize,
                                                       gauss3D_calib(psf),
                                                       kz,
                                                       alpha)
        crlb_ts[i] = compute_crlb(expval_ts,jac_ts)[0]*1000
        
        # Calculate improvement factor
        crlb_if[i] = crlb_astig[i,2] / crlb_ts[i,2]
        
    return(zrange, crlb_astig[:,2], crlb_ts[:,2], crlb_if[:,0])

# Plots an image of the ROI for debuggin
#------------------------------------------------------------------
def plot_rois():
    roisize = 16
    theta = np.array([
        [roisize/2,roisize/2,-0.1,1,0],
        [roisize/2,roisize/2,   0,1,0],
        [roisize/2,roisize/2, 0.1,1,0]
        ])
    ev, jac = compute_derivatives_XYZIBg(theta, roisize, gauss3D_calib(15))
    fig,ax=plt.subplots(1,3); 
    for i in range(len(ax)):
        ax[i].imshow(ev[i])
     
########################################################################
# Demo: 2 angles, NA = 1.33, 1.49, alpha = 1
#-----------------------------------------------------------------------
if __name__ == "__main__":
    
    zrange = 0.3
    fp=0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,7))
    ax1.set_xlabel("z (nm)", fontsize=14)
    ax1.set_ylabel("CRLB z (nm)", fontsize=14)
    ax2.set_xlabel("z (nm)", fontsize=14)
    ax2.set_ylabel("Improvement factor",fontsize=14)
    ax1.set_xlim([0, zrange])
    ax1.set_xticks(np.linspace(0,zrange,7))
    ax1.set_xticklabels([0,50,100,150,200,250,300])
    ax1.set_ylim([0, 20])
    ax1.tick_params(axis='both', labelsize=14)
    ax2.set_xlim([0, zrange])
    ax2.set_xticks(np.linspace(0,zrange,7))
    ax2.set_xticklabels([0,50,100,150,200,250,300])
    ax2.set_ylim([1, 3])
    ax2.tick_params(axis='both', labelsize=14)
    ax1.grid()
    ax2.grid()
    
    # crlb_study(z range, NA range, n of cover slip, 
    #            alpha factor, focal plane position, psf type)
    # assumes n of sample is 1.33
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax1.plot(x, b, label = "NA = 1.33, 1.49; \u03B1 = 1")
    ax2.plot(x, c)
    ax1.plot(x, a, label = "astigmatic only", color="black")
    ax1.legend(fontsize=12)
    plt.show()
    
# Manuscript figure generation
#-----------------------------------------------------------------------
def generate_fig2():
    
    zrange = 0.3
    fp = 0
    
    fig2 = plt.figure(figsize=(10,10))
    
    gs2 = fig2.add_gridspec(3, 2, figure=fig2)
    ax2a = fig2.add_subplot(gs2[:,0])
    ax2b = fig2.add_subplot(gs2[0,1])
    ax2c = fig2.add_subplot(gs2[1,1])
    ax2d = fig2.add_subplot(gs2[2,1])
    fig2.subplots_adjust(wspace=0.25)
    ax2a.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax2a.set_ylabel("CRLB z (nm)", fontsize=14)
    ax2a.set_xlim([0, zrange])
    ax2a.set_xticks(np.linspace(0,zrange,7))
    ax2a.set_xticklabels([0,50,100,150,200,250,300])
    ax2a.set_ylim([0, 20])
    ax2a.yaxis.set_ticks(np.arange(0, 21, 2))
    ax2a.tick_params(axis='both', labelsize=14)
    ax2a.grid()
    ax2a.text(-0.06,20, "a)", fontsize=18)

    ax2d.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax2b.set_ylabel("Improvement factor", fontsize=14)
    ax2c.set_ylabel("Improvement factor", fontsize=14)
    ax2d.set_ylabel("Improvement factor", fontsize=14)
    ax2b.set_xlim([0, zrange])
    ax2b.set_xticks(np.linspace(0,zrange,7))
    ax2b.set_xticklabels([0,50,100,150,200,250,300])
    ax2b.set_ylim([1, 3.3])
    ax2c.set_xlim([0, zrange])
    ax2c.set_xticks(np.linspace(0,zrange,7))
    ax2c.set_xticklabels([0,50,100,150,200,250,300])
    ax2c.set_ylim([1, 3.3])
    ax2d.set_xlim([0, zrange])
    ax2d.set_xticks(np.linspace(0,zrange,7))
    ax2d.set_xticklabels([0,50,100,150,200,250,300])
    ax2d.set_ylim([1, 3.3])
    ax2b.tick_params(axis='y', labelsize=14)
    ax2b.tick_params(axis='x', labelbottom=False)
    ax2c.tick_params(axis='y', labelsize=14)
    ax2c.tick_params(axis='x', labelbottom=False)
    ax2d.tick_params(axis='both', labelsize=14)
    ax2b.grid()
    ax2c.grid()
    ax2d.grid()
    ax2b.text(-0.05,3.3, "b)", fontsize=18)
    ax2c.text(-0.05,3.3, "c)", fontsize=18)
    ax2d.text(-0.05,3.3, "d)", fontsize=18)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp)
    ax2a.plot(x, a, label = "astigmatic only", color='black', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax2a.plot(x, b, label = "2 angles", color = (0.29,0,0.51,1.0), linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.41, 1.49]), 1.51, 1, fp)
    ax2a.plot(x, b, label = "3 angles", color = (0.29,0,0.51,0.8), linestyle='dashed', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.38, 1.44, 1.49]), 1.51, 1, fp)
    ax2a.plot(x, b, label = "4 angles", color = (0.29,0,0.51,0.6), linestyle='-.', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.37, 1.41, 1.45, 1.49]), 1.51, 1, fp)
    ax2a.plot(x, b, label = "5 angles", color = (0.29,0,0.51,0.4), linestyle='dotted', linewidth=3)      
    ax2a.legend(fontsize=14)

    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax2b.plot(x, c, label = "NA=1.33,1.49", color='black', linewidth=2) 
    x, a, b, c = crlb_study(zrange, np.array([1.34, 1.49]), 1.51, 1, fp)
    ax2b.plot(x, c, label = "NA=1.34,1.49", color='#ad5209', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.35, 1.49]), 1.51, 1, fp)
    ax2b.plot(x, c, label = "NA=1.35,1.49", color='#f7a563', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.48]), 1.51, 1, fp)
    ax2b.plot(x, c, label = "NA=1.33,1.48", color='#ad5209', linestyle='dashed', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.47]), 1.51, 1, fp)
    ax2b.plot(x, c, label = "NA=1.33,1.47", color='#f7a563', linestyle='dashed', linewidth=2)
    ax2b.legend(fontsize=9)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax2c.plot(x, c, label = "\u03B1=1", color='black', linewidth=2) 
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 0.95, fp)
    ax2c.plot(x, c, label = "\u03B1=0.95", color='#4a7308', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 0.9, fp)
    ax2c.plot(x, c, label = "\u03B1=0.9", color='#6ba510', linestyle='dashed', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 0.85, fp)
    ax2c.plot(x, c, label = "\u03B1=0.85", color='#6ba510', linestyle='-.', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 0.8, fp)
    ax2c.plot(x, c, label = "\u03B1=0.8", color='#6ba510', linestyle='dotted', linewidth=2)
    ax2c.legend(fontsize=9)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax2d.plot(x, c, label = "NA=1.33,1.49;\u03B1=1", color='black', linewidth=2) 
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.7]), 1.78, 1, fp, psf=17)
    ax2d.plot(x, c, label = "NA=1.33,1.7;\u03B1=1", color='#006bad', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.332, 1.69]), 1.78, 1, fp, psf=17)
    ax2d.plot(x, c, label = "NA=1.332,1.69;\u03B1=1", color='#4abdff', linestyle='dashed', linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.332, 1.69]), 1.78, 0.9, fp, psf=17)
    ax2d.plot(x, c, label = "NA=1.332,1.69;\n\u03B1=0.95", color='#84d6ff', linestyle='dotted', linewidth=3)
    ax2d.legend(fontsize=9)
    
    fig2.savefig('Fig2.eps', bbox_inches='tight', format='eps')
    
def generate_fig3():
    
    zrange = 0.3
    fp = 0
    
    fig3 = plt.figure(figsize=(10,10))
    gs3 = fig3.add_gridspec(3, 2, figure=fig3)
    ax3a = fig3.add_subplot(gs3[:,0])
    ax3b = fig3.add_subplot(gs3[:-1,1])
    ax3c = fig3.add_subplot(gs3[-1,1])
    fig3.subplots_adjust(wspace=0.25)

    ax3a.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax3a.set_ylabel("CRLB z (nm)", fontsize=14)
    ax3a.set_xlim([0, zrange])
    ax3a.set_xticks(np.linspace(0,zrange,7))
    ax3a.set_xticklabels([0,50,100,150,200,250,300])
    ax3a.set_ylim([0, 20])
    ax3a.yaxis.set_ticks(np.arange(0, 21, 2))
    ax3a.tick_params(axis='both', labelsize=14)
    ax3a.grid()
    ax3a.text(-0.06,19.8, "a)", fontsize=18)
    
    ax3c.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax3b.set_ylabel("Improvement factor", fontsize=14)
    ax3c.set_ylabel("Improvement factor", fontsize=14)
    ax3b.set_xlim([0, zrange])
    ax3b.set_xticks(np.linspace(0,zrange,7))
    ax3b.set_xticklabels([0,50,100,150,200,250,300])
    ax3b.set_ylim([1, 6.3])
    ax3c.set_xlim([0, zrange])
    ax3c.set_xticks(np.linspace(0,zrange,7))
    ax3c.set_xticklabels([0,50,100,150,200,250,300])
    ax3c.set_ylim([1, 6.3])
    ax3b.yaxis.set_ticks(np.arange(1, 6.0, 0.5))
    ax3c.yaxis.set_ticks(np.arange(1, 6.0, 1.0))
    ax3b.tick_params(axis='y', labelsize=14)
    ax3b.tick_params(axis='x', labelbottom=False)
    ax3c.tick_params(axis='both', labelsize=14)
    ax3b.grid()
    ax3c.grid()
    ax3b.text(-0.05,6.2, "b)", fontsize=18)
    ax3c.text(-0.05,6.2, "c)", fontsize=18)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp)
    ax3a.plot(x, a, label = "astigmatic only", color='black', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp)
    ax3a.plot(x, b, label = "2 angles, obj. NA=1.49", color='red')
    ax3b.plot(x, c, label = "2 angles, obj. NA=1.49", color='red')
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.51]), 1.51, 1, fp)
    ax3a.plot(x, b, label = "2 angles, glass", color = '#424242')
    ax3b.plot(x, c, label = "2 angles, glass", color = '#424242')
    ax3c.plot(x, c, label = "Glass, \u03B1=1.0", color = '#000000')
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.42, 1.51]), 1.51, 1, fp)
    ax3a.plot(x, b, label = "3 angles, glass", color = '#848484')
    ax3b.plot(x, c, label = "3 angles, glass", color = '#848484')
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.39, 1.45, 1.51]), 1.51, 1, fp)
    ax3b.plot(x, c, label = "4 angles, glass", color = '#c3c3c3')
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.51]), 1.51, 0.95, fp)
    ax3c.plot(x, c, label = "Glass, \u03B1=0.95", color = '#000000', linestyle="dashed")
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.78]), 1.78, 1, fp)
    ax3a.plot(x, b, label = "2 angles, $Al_2O_3$", color = '#4a0094', linestyle="dashed")
    ax3b.plot(x, c, label = "2 angles, $Al_2O_3$", color = '#4a0094', linestyle="dashed")
    ax3c.plot(x, c, label = "$Al_2O_3$, \u03B1=1.0", color = '#3f008d')
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.555, 1.78]), 1.78, 1, fp)
    ax3a.plot(x, b, label = "3 angles, $Al_2O_3$", color = '#9c39ff', linestyle="dashed")
    ax3b.plot(x, c, label = "3 angles, $Al_2O_3$", color = '#9c39ff', linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.48, 1.63, 1.78]), 1.78, 1, fp)
    ax3b.plot(x, c, label = "4 angles, $Al_2O_3$", color = '#be76ef', linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.78]), 1.78, 0.95, fp)
    ax3c.plot(x, c, label = "$Al_2O_3$, \u03B1=0.95", color = '#962dff', linestyle="dashed")
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 2.5]), 2.5, 1, fp)
    ax3a.plot(x, b, label = "2 angles, $TiO_2$", color = '#0018bd', linestyle="dotted", linewidth=2)
    ax3b.plot(x, c, label = "2 angles, $TiO_2$", color = '#0018bd', linestyle="dotted", linewidth=2)
    ax3c.plot(x, c, label = "$TiO_2$, \u03B1=1.0", color = '#000cb9')
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.915, 2.5]), 2.5, 1, fp)
    ax3a.plot(x, b, label = "3 angles, $TiO_2$", color = '#5a73ff', linestyle="dotted", linewidth=2)
    ax3b.plot(x, c, label = "3 angles, $TiO_2$", color = '#5a73ff', linestyle="dotted", linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.72, 2.11, 2.5]), 2.5, 1, fp)
    ax3b.plot(x, c, label = "4 angles, $TiO_2$", color = '#8c9cff', linestyle="dotted", linewidth=2)
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 2.5]), 2.5, 0.95, fp)
    ax3c.plot(x, c, label = "$TiO_2$, \u03B1=0.95", color = '#6179ff', linestyle="dashed")

    ax3a.legend(fontsize=9)
    ax3b.legend(fontsize=9)
    ax3c.legend(fontsize=9)
    
    fig3.savefig('Fig3.eps', bbox_inches='tight', format='eps')
    
def generate_fig4():
        
    zrange = 1

    fig4 = plt.figure(figsize=(10,10))
    gs4 = fig4.add_gridspec(2, 2, figure=fig4)
    ax4a = fig4.add_subplot(gs4[0,:])
    ax4b = fig4.add_subplot(gs4[1,0])
    ax4c = fig4.add_subplot(gs4[1,1])
    fig4.subplots_adjust(hspace=0.25)
    
    ax4a.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax4a.set_ylabel("CRLB z (nm)", fontsize=14)
    ax4a.set_xlim([0, zrange])
    ax4a.set_xticks(np.linspace(0,zrange,11))
    ax4a.set_xticklabels([0,100,200,300,400,500,600,700,800,900,1000])
    ax4a.set_ylim([0, 20])
    ax4a.yaxis.set_ticks(np.arange(0, 41, 5))
    ax4a.tick_params(axis='both', labelsize=12)
    ax4a.grid()
    ax4a.text(-0.1,40.2, "a)", fontsize=18)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0)
    ax4a.plot(x, a, label = "astigmatic only; focal plane at 0 nm", color='#B9167A', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.15)
    ax4a.plot(x, a, label = "astigmatic only; focal plane at 150 nm", color='#FF938A', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.3)
    ax4a.plot(x, a, label = "astigmatic only; focal plane at 300 nm", color='#FECFAA', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.75)
    ax4a.plot(x, a, label = "astigmatic only; focal plane at 750 nm", color='#bebebe', linewidth=3)

    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0)
    ax4a.plot(x, b, label = "2 angles; focal plane at 0 nm", color = '#B9167A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.15)
    ax4a.plot(x, b, label = "2 angles; focal plane at 150 nm", color = '#FF938A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.3)
    ax4a.plot(x, b, label = "2 angles; focal plane at 300 nm", color = '#FECFAA', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.75)
    ax4a.plot(x, b, label = "2 angles; focal plane at 750 nm", color = '#bebebe', linewidth=3, linestyle="dashed")
    
    zrange = 0.3

    ax4b.set_xlabel(r"$\theta_\mathrm{z}$ (nm)", fontsize=14)
    ax4b.set_ylabel("CRLB z (nm)", fontsize=14)
    ax4b.set_xlim([0, zrange])
    ax4b.set_xticks(np.linspace(0,zrange,7))
    ax4b.set_xticklabels([0,50,100,150,200,250,300])
    ax4b.set_ylim([0, 20])
    ax4b.yaxis.set_ticks(np.arange(0, 21, 2))
    ax4b.tick_params(axis='both', labelsize=12)
    ax4b.grid()
    ax4b.text(-0.06,20, "b)", fontsize=18)
    ax4c.set_xlabel("z (nm)", fontsize=14)
    ax4c.set_ylabel("Improvement factor",fontsize=14)    
    ax4c.set_xlim([0, zrange])
    ax4c.set_xticks(np.linspace(0,zrange,7))
    ax4c.set_xticklabels([0,50,100,150,200,250,300])
    ax4c.set_ylim([1, 3.3])
    ax4c.tick_params(axis='both', labelsize=12)
    ax4c.grid()
    ax4c.text(-0.05,3.3, "c)", fontsize=18)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0)
    ax4b.plot(x, a, label = "astigmatic only; focal plane at 0 nm", color='#B9167A', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.15)
    ax4b.plot(x, a, label = "astigmatic only; focal plane at 150 nm", color='#FF938A', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.3)
    ax4b.plot(x, a, label = "astigmatic only; focal plane at 300 nm", color='#FECFAA', linewidth=3)
    x, a, b, c = crlb_study(zrange, np.array([1.3301]), 1.51, 1, fp=0.75)
    ax4b.plot(x, a, label = "astigmatic only; focal plane at 750 nm", color='#bebebe', linewidth=3)
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0)
    ax4b.plot(x, b, label = "2 angles; focal plane at 0 nm", color = '#B9167A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.15)
    ax4b.plot(x, b, label = "2 angles; focal plane at 150 nm", color = '#FF938A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.3)
    ax4b.plot(x, b, label = "2 angles; focal plane at 300 nm", color = '#FECFAA', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.75)
    ax4b.plot(x, b, label = "2 angles; focal plane at 750 nm", color = '#bebebe', linewidth=3, linestyle="dashed")
    
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0)
    ax4c.plot(x, c, label = "2 angles; focal plane at 0 nm", color = '#B9167A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.15)
    ax4c.plot(x, c, label = "2 angles; focal plane at 150 nm", color = '#FF938A', linewidth=3, linestyle="dashed")
    x, a, b, c = crlb_study(zrange, np.array([1.3301, 1.49]), 1.51, 1, fp=0.3)
    ax4c.plot(x, c, label = "2 angles; focal plane at 300 nm", color = '#FECFAA', linewidth=3, linestyle="dashed")
    
    ax4a.legend(fontsize=12, bbox_to_anchor=(0.5, -1.92), loc="lower center", title="NA = 1.33, 1.49; \u03B1 = 1.0", borderpad=1, ncol=2)
    
    fig4.savefig('Fig4.eps', bbox_inches='tight', format='eps')