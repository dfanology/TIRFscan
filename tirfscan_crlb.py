# -*- coding: utf-8 -*-
"""
CRLB calculation for TIRFscan
@authors: jelmer, daniel, carlas
"""
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt

####################
# Sets up the imaging model
#----------------------------------------------------

# Calibration data for astigmatic imaging
#----------------------------------------
gauss3D_calib = [
    [1.0573989152908325,
     -0.14864186942577362,
     0.1914146989583969,
     0.10000000149011612],
     [1.0528310537338257,
     0.14878079295158386,
     0.18713828921318054,
     9.999999974752427e-07]]

# Calculation of mu and first derivative for astigmatic imaging
#--------------------------------------------------------------    
def mujac_XYZIBg(theta, numpixels, calib):
    
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
    # Pixel centers (mu and dmu, equations 3 and 4)
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
def mujac_ts_XYZIBg(theta, numpixels, calib, kz, alpha):
    
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
        mujac_XYZIBg(theta_psf,numpixels,calib))
    # Different backgrounds for each TIRF angle (scaled value)
    bg_per_angle = (kz[0]/kz)/np.sum(kz[0]/kz)
    bg_per_angle = bg_per_angle[None,:,None,None]
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
def crlb_study(zmax, numangles, minNA, maxNA, ncovglass, alpha):
    
    # Generally these never change
    roisize = 16
    nwater = 1.33 
    ill_lambda = .640
    I= 1000
    bg = 10
    k0=2*np.pi/ill_lambda
    zrange = np.linspace(0,zmax,int(zmax*1000))
    npos = len(zrange)
    
    # Set the TIRF angles
    NARange = np.linspace(minNA,maxNA,numangles)
    #if j ==2: NARange = np.array([1.3301,1.45, 1.49])
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
        expval,jac=mujac_XYZIBg(th,roisize,gauss3D_calib)
        
        # Calculate CRLB for just astigmatic imaging
        crlb_astig[i] = compute_crlb(expval,jac)[0]*1000
        
        # Calculate CRLB for tirfscan with astigmatic imaging
        exc[i,:] = alpha*excitation_z(th, kz) + 1 -alpha #alpha here
        th[:,3] = I/ np.sum(exc[i])
        expval_ts,jac_ts=mujac_ts_XYZIBg(th,roisize,gauss3D_calib,kz,alpha)
        crlb_ts[i] = compute_crlb(expval_ts,jac_ts)[0]*1000
        
        # Calculate improvement factor
        crlb_if[i] = crlb_astig[i,2] / crlb_ts[i,2]
        
    return(zrange, crlb_astig[:,2], crlb_ts[:,2], crlb_if[:,0])
     
########################################################################
# Demo: 3 angles, NA = 1.33, 1.41, 1.49, alpha = 1, 0.95, 0.9, 0.85, 0.8
#-----------------------------------------------------------------------
if __name__ == "__main__":
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,7))
    ax1.set_xlabel("z (nm)", fontsize=14)
    ax1.set_ylabel("CRLB z (nm)", fontsize=14)
    ax2.set_xlabel("z (nm)", fontsize=14)
    ax2.set_ylabel("Improvement factor",fontsize=14)
    ax1.set_xlim([0, 0.3])
    ax1.set_ylim([0, 20])
    ax1.tick_params(axis='both', labelsize=14)
    ax2.set_xlim([0, 0.3])
    ax2.set_ylim([1, 3])
    ax2.tick_params(axis='both', labelsize=14)
    ax1.grid()
    ax2.grid()
    
    # crlb_study(z range, number of angles, minimum NA,
    #               maximum NA, n of cover slip, alpha factor)
    # assumes n of sample is 1.33
    
    nang = 3 #to get figure 2c (i.e., 2 angles), change this to 2
    
    x, a, b, c = crlb_study(0.3, nang, 1.3301, 1.49, 1.51, 1)
    ax1.plot(x, a, label = "astigmatic only", color='black')
    
    for i in [1, 0.95, 0.9, 0.85, 0.8]:
        x, a, b, c = crlb_study(0.3, nang, 1.3301, 1.49, 1.51, i)
        ax1.plot(x, b, label = f"alpha = {i}")
        ax2.plot(x, c)
    
    fig.suptitle(f'TIRFscan with {nang} angles from NA = 1.33 to 1.49', 
                 fontsize=16)
    ax1.legend(fontsize=12)
    plt.show()


    
    