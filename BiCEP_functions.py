import pandas as pd
from importlib import reload # allows reloading of modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipywidgets as widgets
from IPython.display import display, clear_output
from importlib import reload
import asyncio
import pickle
import pmagpy.pmag as pmag
import pmagpy.ipmag as ipmag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pmagpy import contribution_builder as cb
import scipy as scipy
import pickle
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

#The sortarai function from pmagpy, this will soon be modified so that additivity checks work.
model_circle_fast=pickle.load(open('model_circle_fast.pkl','rb'))
model_circle_slow=pickle.load(open('model_circle_slow.pkl','rb'))

def get_mad(IZZI):
    """Calculates the free Maximum Angle of Deviation (MAD) of Kirshvink et al (1980)"""
    pca=PCA(n_components=3)
    fit=pca.fit(IZZI.loc[:,'NRM_x':'NRM_z'].values).explained_variance_
    MAD=np.degrees(np.arctan(np.sqrt((fit[2]+fit[1])/(fit[0]))))
    return MAD

def TaubinSVD(x,y):
    """
    Function from PmagPy
    algebraic circle fit
    input: list [[x_1, y_1], [x_2, y_2], ....]
    output: a, b, r.  a and b are the center of the fitting circle, and r is the radius

     Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """
    X = np.array(list(map(float, x)))
    Xprime=X
    Y = np.array(list(map(float, y)))
    Yprime=Y
    XY = np.array(list(zip(X, Y)))
    XY = np.array(XY)
    X = XY[:,0] - np.mean(XY[:,0]) # norming points by x avg
    Y = XY[:,1] - np.mean(XY[:,1]) # norming points by y avg
    centroid = [np.mean(XY[:,0]), np.mean(XY[:,1])]
    Z = X * X + Y * Y
    Zmean = np.mean(Z)
    Z0 = (Z - Zmean)/(2. * np.sqrt(Zmean))
    ZXY = np.array([Z0, X, Y]).T
    U, S, V = np.linalg.svd(ZXY, full_matrices=False) #
    V = V.transpose()
    A = V[:,2]
    A[0] = A[0]/(2. * np.sqrt(Zmean))
    A = np.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
    a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
    r = np.sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/abs(A[0])/2
    errors=[]
    for i in list(range(0,len(Xprime)-1)):
        errors.append((np.sqrt((Xprime[i]-a)**2+(Yprime[i]-b)**2)-r)**2)
    sigma=np.sqrt((sum(errors))/(len(Xprime)-1))


    return a,b,r,sigma

def get_drat(IZZI,IZZI_trunc,P):
    """Calculates the difference ratio (DRAT) of pTRM checks
    (Selkin and Tauxe, 2000) to check for alteration"""
    IZZI_reduced=IZZI[IZZI.temp_step.isin(P.temp_step)]
    a=np.sum((IZZI_trunc.PTRM-np.mean(IZZI_trunc.PTRM))*(IZZI_trunc.NRM-np.mean(IZZI_trunc.NRM)))
    b=a/np.abs(a)*np.sqrt(np.sum((IZZI_trunc.NRM-np.mean(IZZI_trunc.NRM))**2)/np.sum((IZZI_trunc.PTRM-np.mean(IZZI_trunc.PTRM))**2))
    yint=np.mean(IZZI_trunc.NRM)-b*np.mean(IZZI_trunc.PTRM)
    line={'slope':b,'intercept':yint}

    xprime=0.5*(IZZI_trunc.PTRM+(IZZI_trunc.NRM-line['intercept'])/line['slope'])
    yprime=0.5*(IZZI_trunc.NRM+line['slope']*IZZI_trunc.PTRM+line['intercept'])
    scalefactor=np.sqrt((min(xprime)-max(xprime))**2+(min(yprime)-max(yprime))**2)
    absdiff=max(np.abs(P.PTRM.values-IZZI_reduced.PTRM.values)/scalefactor)*100
    return(absdiff)

def calculate_anisotropy_correction(IZZI):
    """Calculates anisotropy correction factor for a
    paleointensity interpretation, given an s tensor"""

    #Convert the s tensor into a numpy array
    strlist=IZZI['s_tensor'].iloc[0].split(':')
    slist=[]
    for stringo in strlist:
        slist.append(float(stringo.strip()))
    stensor=np.array([[slist[0],slist[3],slist[5]],[slist[3],slist[1],slist[4]],[slist[5],slist[4],slist[2]]])

    #Fit a PCA to the IZZI directions
    NRM_trunc_dirs=IZZI.loc[:,'NRM_x':'NRM_z']
    pca=PCA(n_components=3)
    pca=pca.fit(NRM_trunc_dirs)

    #Calculate the anisotropy correction factor (see Standard Paleointensity Definitions)
    vector=pca.components_[0]
    vector=vector/np.sqrt(np.sum(vector**2))
    ancvector=np.matmul(np.linalg.inv(stensor),vector)
    ancvector=ancvector/np.sqrt(np.sum(ancvector**2))
    labmag=np.matmul(stensor,np.array([0,0,-1]))
    ancmag=np.matmul(stensor,ancvector)
    c=np.sqrt(np.sum(labmag**2))/np.sqrt(np.sum(ancmag**2))
    return(c)

def calculate_NLT_correction(IZZI,c):
    """Calculates the correction for non linear TRM for a paleointensity interpretation, given the anisotropy and cooling rate corrections"""
    a=np.sum((IZZI.PTRM-np.mean(IZZI.PTRM))*(IZZI.NRM-np.mean(IZZI.NRM)))
    b=a/np.abs(a)*np.sqrt(np.sum((IZZI.NRM-np.mean(IZZI.NRM))**2)/np.sum((IZZI.PTRM-np.mean(IZZI.PTRM))**2))
    beta=IZZI['NLT_beta'].iloc[0]
    correction=c*IZZI.correction.iloc[0]
    B_lab=IZZI.B_lab.iloc[0]*1e6
    total_correction=(np.arctanh(correction*np.abs(b)*np.tanh(beta*B_lab)))/(np.abs(b)*beta*B_lab)
    return(total_correction)

def prep_data_for_fitting(IZZI_filtered,IZZI_original):
    """Returns the needed data for a paleointensity interpretation to perform the BiCEP method (Cych et al, in prep.), calculates all corrections for a specimen"""

    specimen=IZZI_original.specimen.iloc[0] #Specimen name
    methcodes='' #String For Method Codes
    extracolumnsdict={} #Extra Column Information for MagIC export (corrections)

    #Calculate Anisotropy Correction:
    if len(IZZI_original.dropna(subset=['s_tensor']))>0:
        c=calculate_anisotropy_correction(IZZI_filtered)
        extracolumnsdict['int_corr_aniso']=c
        #Get method code depending on anisotropy type (AARM or ATRM)
        methcodes+=IZZI_original['aniso_type'].iloc[0]
    else:
        c=1

    #Get Cooling Rate Correction
    if IZZI_original.correction.iloc[0]!=1:
        methcodes+=':DA-CR-TRM' #method code for cooling rate correction
        extracolumnsdict['int_corr_cooling_rate']=IZZI_original.correction.iloc[0]


    #Calculate nonlinear TRM Correction
    if len(IZZI_original.dropna(subset=['NLT_beta']))>0:
        methcodes+=':DA-NL' #method code for nonlinear TRM correction
        total_correction=calculate_NLT_correction(IZZI_filtered,c) #total correction (combination of all three corrections)
        extracolumnsdict['int_corr_nlt']=total_correction/(c*IZZI_original.correction.iloc[0]) #NLT correction is total correction/original correction.
    else:
        total_correction=c*IZZI_original.correction.iloc[0]

    #Converting Arai plot data to useable form
    NRM0=IZZI_original.NRM.iloc[0]
    NRMS=IZZI_filtered.NRM.values/NRM0
    PTRMS=IZZI_filtered.PTRM.values/NRM0/total_correction #We divide our pTRMs by the total correction, because we scale the pTRM values so that the maximum pTRM is one, this doesn't affect the fit and just gets scaled back when converting the circle tangent slopes back to intensities as would be expected, but it's easier to apply this here.

    PTRMmax=max(IZZI_original.PTRM/NRM0/total_correction) #We scale by our maximum pTRM to perform the circle fit.
    line=bestfit_line(IZZI_original.PTRM/NRM0/total_correction,IZZI_original.NRM/NRM0) #best fitting line to the pTRMs
    scale=np.sqrt((line['intercept']/line['slope'])**2+(line['intercept'])**2) #Flag- is this ever used?

    PTRMS=PTRMS/PTRMmax  #Scales the pTRMs so the maximum pTRM is one

    #We subtract the minimum pTRM and NRM to maintain aspect ratio and make circle fitting easier.
    minPTRM=min(PTRMS)
    minNRM=min(NRMS)
    PTRMS=PTRMS-minPTRM
    NRMS=NRMS-minNRM

    #We perform the Taubin least squares circle fit to get values close to the Bayesian maximum likelihood to initialize our MCMC sampler at, this makes sampling a lot easier than initializing at a random point (which may have infinitely low probability).

    x_c,y_c,R,sigma=TaubinSVD(PTRMS,NRMS) #Calculate x_c,y_c and R
    dist_to_edge=abs(np.sqrt(x_c**2+y_c**2)-R) #Calculate D (dist_to_edge)
    phi=np.radians(np.degrees(np.arctan(y_c/x_c))%180)
    #Calculate (and ensure the sign of) k
    if y_c<0:
        k=-1/R
    else:
        k=1/R

    B_lab=IZZI_filtered.B_lab.unique()[0]*1e6

    return(scale,minPTRM,minNRM,PTRMmax,k,phi,dist_to_edge,sigma,PTRMS,NRMS,B_lab,methcodes,extracolumnsdict)


def BiCEP_fit(specimenlist,temperatures=None,n_samples=30000,priorstd=20,model=None,**kwargs):
    minPTRMs=[]
    minNRMs=[]
    IZZI_list=[]
    B_lab_list=[]
    klist=[]
    NRM0s=[]
    pTRMsList=np.array([])
    NRMsList=np.array([])
    lengths=[]
    philist=[]
    dist_to_edgelist=[]
    B_ancs=[]
    dmaxlist=[]
    PTRMmaxlist=[]
    centroidlist=[]
    spec_old=''
    newmethcodes={}
    newcolumns={}
    i=0
    for specimen in specimenlist:
        if spec_old==specimen:
            i+=1
        else:
            i=0
        spec_old=specimen
        IZZI_original=temps[(temps.specimen==specimen)&((temps.steptype=='IZ')|(temps.steptype=="ZI"))]
        if temperatures==None:
            IZZI_filtered=IZZI_original
        else:
            IZZI_filtered=IZZI_original[(IZZI_original.temp_step>=temperatures[specimen][i,0])&(IZZI_original.temp_step<=temperatures[specimen][i,1])]


        scale,minPTRM,minNRM,PTRMmax,k,phi,dist_to_edge,sigma,PTRMS,NRMS,B_lab,methcodestr,extracolumnsdict=prep_data_for_fitting(IZZI_filtered,IZZI_original)
        newcolumns[specimen]=extracolumnsdict
        newmethcodes[specimen]=methcodestr

        if len(IZZI_filtered)<=3:
            print('Specimen Rejected- Too Few Points to make an interpretation')
        NRM0=IZZI_filtered.NRM.iloc[0]
        minPTRMs.append(minPTRM)
        minNRMs.append(minNRM)
        line=bestfit_line(IZZI_filtered.PTRM,IZZI_filtered.NRM)
        B_anc=-line['slope']*B_lab*IZZI_filtered.correction.iloc[0]
        B_ancs.append(B_anc)
        Pi,Pj=np.meshgrid(PTRMS,PTRMS)
        Ni,Nj=np.meshgrid(NRMS,NRMS)
        dmax=np.amax(np.sqrt((Pi-Pj)**2+(Ni-Nj)**2))
        centroid=np.sqrt(np.mean(PTRMS)**2+np.mean(NRMS)**2)
        IZZI_list.append(IZZI_filtered)
        B_lab_list.append(B_lab)
        klist.append(k)
        philist.append(phi)
        dist_to_edgelist.append(dist_to_edge)
        NRM0s.append(NRM0)
        pTRMsList=np.append(pTRMsList,PTRMS)
        NRMsList=np.append(NRMsList,NRMS)
        lengths.append(int(len(PTRMS)))
        dmaxlist.append(dmax)
        PTRMmaxlist.append(PTRMmax)
        centroidlist.append(centroid)
    if model==None:
        if len(specimenlist)<7:
            model_circle=model_circle_slow
        else:
            model_circle=model_circle_fast
    else:
        model_circle=model
    fit_circle=model_circle.sampling (
        data={'I':len(pTRMsList),'M':len(lengths),'PTRM':pTRMsList,'NRM':NRMsList,'N':lengths,'PTRMmax':PTRMmaxlist,'B_labs':B_lab_list,'dmax':np.sqrt(dmaxlist),'centroid':centroidlist,'priorstd':priorstd},iter=n_samples,warmup=int(n_samples/2),
        init=[{'k_scale':np.array(klist)*np.array(dist_to_edgelist),'phi':philist,'dist_to_edge':dist_to_edgelist,'int_real':B_ancs}]*4,**kwargs)

    return(fit_circle,newmethcodes,newcolumns)

def sufficient_statistics(ptrm, nrm):
    """
    inputs list of ptrm and nrm data and computes sufficent statistcs needed
    for computations
    """

    corr = np.cov( np.stack((ptrm, nrm), axis=0) )

    return {'xbar': np.mean(ptrm), 'ybar': np.mean(nrm), 'S2xx': corr[0,0], 'S2yy': corr[1,1], 'S2xy': corr[0,1] }

def bestfit_line(ptrm, nrm):
    """
    returns the slope and intercept of the best fit line to a set of points with NRM and PTRM using Bayesian maximum likelihood estimate
    """
    stat = sufficient_statistics(ptrm, nrm)

    w = .5*(stat['S2xx'] - stat['S2yy'])/stat['S2xy']
    m = -w-np.sqrt(w**2+1)
    b = stat['ybar']-m*stat['xbar']

    return {'slope': m, 'intercept': b }

def sortarai(datablock, s, Zdiff, **kwargs):
    """
     sorts data block in to first_Z, first_I, etc.

    Parameters
    _________
    datablock : Pandas DataFrame with Thellier-Tellier type data
    s : specimen name
    Zdiff : if True, take difference in Z values instead of vector difference
            NB:  this should always be False
    **kwargs :
        version : data model.  if not 3, assume data model = 2.5

    Returns
    _______
    araiblock : [first_Z, first_I, ptrm_check,
                 ptrm_tail, zptrm_check, GammaChecks]
    field : lab field (in tesla)
    """
    if 'version' in list(kwargs.keys()) and kwargs['version'] == 3:
        dec_key, inc_key, csd_key = 'dir_dec', 'dir_inc', 'dir_csd'
        Mkeys = ['magn_moment', 'magn_volume', 'magn_mass', 'magnitude','dir_csd']
        meth_key = 'method_codes'
        temp_key, dc_key = 'treat_temp', 'treat_dc_field'
        dc_theta_key, dc_phi_key = 'treat_dc_field_theta', 'treat_dc_field_phi'
        # convert dataframe to list of dictionaries
        datablock = datablock.to_dict('records')
    else:
        dec_key, inc_key, csd_key = 'measurement_dec', 'measurement_inc','measurement_csd'
        Mkeys = ['measurement_magn_moment', 'measurement_magn_volume',
                 'measurement_magn_mass', 'measurement_magnitude']
        meth_key = 'magic_method_codes'
        temp_key, dc_key = 'treatment_temp', 'treatment_dc_field'
        dc_theta_key, dc_phi_key = 'treatment_dc_field_theta', 'treatment_dc_field_phi'
    first_Z, first_I, zptrm_check, ptrm_check, ptrm_tail = [], [], [], [], []
    field, phi, theta = "", "", ""
    starthere = 0
    Treat_I, Treat_Z, Treat_PZ, Treat_PI, Treat_M = [], [], [], [], []
    ISteps, ZSteps, PISteps, PZSteps, MSteps = [], [], [], [], []
    GammaChecks = []  # comparison of pTRM direction acquired and lab field
    rec = datablock[0]
    for key in Mkeys:
        if key in list(rec.keys()) and rec[key] != "":
            momkey = key
            break
# first find all the steps
    for k in range(len(datablock)):
        rec = datablock[k]
        temp = float(rec[temp_key])
        methcodes = []
        tmp = rec[meth_key].split(":")
        for meth in tmp:
            methcodes.append(meth.strip())
        if 'LT-T-I' in methcodes and 'LP-TRM' not in methcodes and 'LP-PI-TRM' in methcodes:
            Treat_I.append(temp)
            ISteps.append(k)
            if field == "":
                field = float(rec[dc_key])
            if phi == "":
                phi = float(rec[dc_phi_key])
                theta = float(rec[dc_theta_key])
# stick  first zero field stuff into first_Z
        if 'LT-NO' in methcodes:
            Treat_Z.append(temp)
            ZSteps.append(k)
        if 'LT-T-Z' in methcodes:
            Treat_Z.append(temp)
            ZSteps.append(k)
        if 'LT-PTRM-Z' in methcodes:
            Treat_PZ.append(temp)
            PZSteps.append(k)
        if 'LT-PTRM-I' in methcodes:
            Treat_PI.append(temp)
            PISteps.append(k)
        if 'LT-PTRM-MD' in methcodes:
            Treat_M.append(temp)
            MSteps.append(k)
        if 'LT-NO' in methcodes:
            dec = float(rec[dec_key])
            inc = float(rec[inc_key])
            str = float(rec[momkey])
            if csd_key not in rec.keys():
                sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            elif rec[csd_key]!=None:
                sig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*str
            else:
                sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            first_I.append([273, 0., 0., 0., 0., 1])
            first_Z.append([273, dec, inc, str, sig, 1])  # NRM step
    for temp in Treat_I:  # look through infield steps and find matching Z step
        if temp in Treat_Z:  # found a match
            istep = ISteps[Treat_I.index(temp)]
            irec = datablock[istep]
            methcodes = []
            tmp = irec[meth_key].split(":")
            for meth in tmp:
                methcodes.append(meth.strip())
            # take last record as baseline to subtract
            brec = datablock[istep - 1]
            zstep = ZSteps[Treat_Z.index(temp)]
            zrec = datablock[zstep]
    # sort out first_Z records
            if "LP-PI-TRM-IZ" in methcodes:
                ZI = 0
            else:
                ZI = 1
            dec = float(zrec[dec_key])
            inc = float(zrec[inc_key])
            str = float(zrec[momkey])
            if csd_key not in rec.keys():
                sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            elif rec[csd_key]!=None:

                sig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*str
            else:
                sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            first_Z.append([temp, dec, inc, str, sig, ZI])

    # sort out first_I records
            idec = float(irec[dec_key])
            iinc = float(irec[inc_key])
            istr = float(irec[momkey])
            X = pmag.dir2cart([idec, iinc, istr])
            BL = pmag.dir2cart([dec, inc, str])
            I = []
            for c in range(3):
                I.append((X[c] - BL[c]))
            if I[2] != 0:
                iDir = pmag.cart2dir(I)
                if csd_key not in rec.keys():
                    isig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
                elif rec[csd_key]!=None:
                    isig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*istr
                else:
                    isig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*istr
                isig = np.sqrt(isig**2+sig**2)

                if Zdiff == 0:
                    first_I.append([temp, iDir[0], iDir[1], iDir[2], isig, ZI])
                else:
                    first_I.append([temp, 0., 0., I[2], 0., isig, ZI])

                gamma = pmag.angle([iDir[0], iDir[1]], [phi, theta])
            else:
                first_I.append([temp, 0., 0., 0., 0., ZI])
                gamma = 0.0
# put in Gamma check (infield trm versus lab field)
            if 180. - gamma < gamma:
                gamma = 180. - gamma
            GammaChecks.append([temp - 273., gamma])
    for temp in Treat_PI:  # look through infield steps and find matching Z step
        step = PISteps[Treat_PI.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])


        brec = datablock[step - 1]  # take last record as baseline to subtract
        pdec = float(brec[dec_key])
        pinc = float(brec[inc_key])
        pint = float(brec[momkey])
        X = pmag.dir2cart([dec, inc, str])
        prevX = pmag.dir2cart([pdec, pinc, pint])
        I = []
        for c in range(3):
            I.append(X[c] - prevX[c])
        dir1 = pmag.cart2dir(I)
        if csd_key not in rec.keys():
            sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            psig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*dir1[2]
        elif rec[csd_key]!=None:
            sig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*str
            psig=np.radians(float(brec[csd_key]))*np.sqrt(3)/np.sqrt(2)*dir1[2]
        else:
            sig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            psig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*dir1[2]
        psig=np.sqrt(sig**2+psig**2)
        if Zdiff == 0:
            ptrm_check.append([temp, dir1[0], dir1[1], dir1[2], sig])
        else:
            ptrm_check.append([temp, 0., 0., I[2]], sig)
# in case there are zero-field pTRM checks (not the SIO way)
    for temp in Treat_PZ:
        step = PZSteps[Treat_PZ.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])
        brec = datablock[step - 1]
        pdec = float(brec[dec_key])
        pinc = float(brec[inc_key])
        pint = float(brec[momkey])
        X = pmag.dir2cart([dec, inc, str])
        prevX = pmag.dir2cart([pdec, pinc, pint])
        I = []
        for c in range(3):
            I.append(X[c] - prevX[c])
        dir2 = pmag.cart2dir(I)
        if csd_key not in rec.keys():
            sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            psig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*dir1[2]
        elif rec[csd_key]!=None:
            sig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*str
            psig= np.radians(float(brec[csd_key]))*np.sqrt(3)/np.sqrt(2)*dir2[2]
        else:
            sig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
            psig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*dir1[2]
        psig=np.sqrt(sig**2+psig**2)
        zptrm_check.append([temp, dir2[0], dir2[1], dir2[2],psig])
    # get pTRM tail checks together -
    for temp in Treat_M:
        # tail check step - just do a difference in magnitude!
        step = MSteps[Treat_M.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])
        if csd_key not in rec.keys():
            sig= np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
        elif rec[csd_key]!=None:
            sig = np.radians(float(rec[csd_key]))*np.sqrt(3)/np.sqrt(2)*str
        else:
            sig = np.radians(2)*np.sqrt(3)/np.sqrt(2)*str
        if temp in Treat_Z:
            step = ZSteps[Treat_Z.index(temp)]
            brec = datablock[step]
            pint = float(brec[momkey])
#        X=dir2cart([dec,inc,str])
#        prevX=dir2cart([pdec,pinc,pint])
#        I=[]
#        for c in range(3):I.append(X[c]-prevX[c])
#        d=cart2dir(I)
#        ptrm_tail.append([temp,d[0],d[1],d[2]])
            # difference - if negative, negative tail!
            ptrm_tail.append([temp, dec, inc, str, sig])
        else:
            print(
                s, '  has a tail check with no first zero field step - check input file! for step', temp - 273.)
#
# final check
#
    if len(first_Z) != len(first_I):
        print(len(first_Z), len(first_I))
        print(" Something wrong with this specimen! Better fix it or delete it ")
        input(" press return to acknowledge message")
    araiblock = (first_Z, first_I, ptrm_check,
                 ptrm_tail, zptrm_check, GammaChecks)
    return araiblock, field

def NLTsolver(fields,a,b):
    """Makes the non linear TRM correction"""
    return(a* np.tanh(b*fields))
def convert_intensity_measurements(measurements):
    """Converts a measurements table with only intensity experiments into the internal data format used by the BiCEP method"""
    specimens=list(measurements.specimen.unique())#This function constructs the 'temps' dataframe (used to plot Arai plots)
    #this may take a while to run depending on the number of specimens.
    #Constructs initial empty 'temps' dataframe
    data_array=np.empty(shape=(16,0))
    for specimen in specimens:
            print('Working on:',specimen)
            try:
                araiblock,field=sortarai(measurements[measurements.specimen==specimen],specimen, Zdiff=False,version=3) #Get arai data
                sitename=measurements[measurements.specimen==specimen].site.unique()
                first_Z,first_I,ptrm_check,ptrm_tail,zptrm_check,GammaChecks=araiblock #Split NRM and PTRM values into step types
                B_lab=np.full(len(first_Z),field) #Lab field used
                m=len(first_Z)
                NRM_dec_max=first_Z[m-1][1]
                NRM_inc_max=first_Z[m-1][2]
                NRM_int_max=first_Z[m-1][3]
                PTRM_dec_max=first_I[m-1][1]
                PTRM_inc_max=first_I[m-1][2]
                PTRM_int_max=first_I[m-1][3]
                NRM_vector_max=pmag.dir2cart([NRM_dec_max,NRM_inc_max,NRM_int_max])
                PTRM_vector_max=pmag.dir2cart([PTRM_dec_max,PTRM_inc_max,PTRM_int_max])
                PTRM_vector_max=PTRM_vector_max-NRM_vector_max
                NRMS=[first_Z[i][3] for i in list(range(len(first_Z)))]
                first_Z=np.array(first_Z)
                first_I=np.array(first_I)


                if min(NRMS)/NRMS[0]<0.25:
                    if(len(first_Z))>1:
                        sample=np.full(len(first_Z),measurements[measurements.specimen==specimen]['sample'].unique()[0]) #Get sample name
                        site=np.full(len(first_Z),measurements[measurements.specimen==specimen].site.unique()[0]) #Get site name
                        specarray=np.full(len(first_Z),specimen)
                        temp_step=first_Z[:,0] #Gets the temperature in kelvin we use
                        NRM=first_Z[:,3] #NRM value (in first_Z dataframe)
                        zbinary=first_Z[:,5]#Is it a ZI or an IZ step?
                        zbinary=zbinary.astype('object')
                        zbinary[zbinary==1]='ZI'
                        zbinary[zbinary==0]='IZ'
                        steptype=zbinary
                        PTRM=first_I[:,3] #PTRM value (in first_I dataframe)
                        PTRM_sigma=first_I[:,4]

                        NRM_dec=first_Z[:,1]
                        NRM_inc=first_Z[:,2]
                        NRM_int=NRM
                        NRM_sigma=first_Z[:,4]

                        NRM_vector=pmag.dir2cart(np.array([NRM_dec,NRM_inc,NRM_int]).T)
                        PTRM_vector=pmag.dir2cart(np.array([first_I[:,1],first_I[:,2],first_I[:,3]]).T)

                        NRM_x=NRM_vector[:,0]
                        NRM_y=NRM_vector[:,1]
                        NRM_z=NRM_vector[:,2]

                        PTRM_x=PTRM_vector[:,0]
                        PTRM_y=PTRM_vector[:,1]
                        PTRM_z=PTRM_vector[:,2]

                        newarray=np.array([specarray,sample,site,NRM,PTRM,NRM_x,NRM_y,NRM_z,PTRM_x,PTRM_y,PTRM_z,NRM_sigma,PTRM_sigma,B_lab,steptype,temp_step])
                        data_array=np.concatenate((data_array,newarray),axis=1)

                        #Doing PTRM Checks Part
                        ptrm_check=np.array(ptrm_check)
                        temp_step=ptrm_check[:,0]
                        smallarray=data_array
                        sample=np.full(len(ptrm_check),measurements[measurements.specimen==specimen]['sample'].unique()[0]) #Get sample name
                        site=np.full(len(ptrm_check),measurements[measurements.specimen==specimen].site.unique()[0]) #Get site name
                        specarray=np.full(len(ptrm_check),specimen)
                        B_lab=np.full(len(ptrm_check),field)
                        PTRM=ptrm_check[:,3]
                        PTRM_sigma=ptrm_check[:,4]
                        intersect=data_array[:,(data_array[0]==specimen)&(np.in1d(data_array[-1].astype('float'),temp_step.astype('float')))]
                        NRM_vector=np.array([intersect[5],intersect[6],intersect[7]])
                        NRM_sigma=intersect[11]
                        PTRM_vector=pmag.dir2cart(np.array([ptrm_check[:,1],ptrm_check[:,2],ptrm_check[:,3]]).T)
                        NRM_x=NRM_vector[0]
                        NRM_y=NRM_vector[1]
                        NRM_z=NRM_vector[2]


                        PTRM_x=PTRM_vector[:,0]
                        PTRM_y=PTRM_vector[:,1]
                        PTRM_z=PTRM_vector[:,2]
                        NRM=intersect[3]
                        steptype=np.full(len(ptrm_check),'P')
                        if len(NRM)==len(PTRM):

                            newarray=np.array([specarray,sample,site,NRM,PTRM,NRM_x,NRM_y,NRM_z,PTRM_x,PTRM_y,PTRM_z,NRM_sigma,PTRM_sigma,B_lab,steptype,temp_step])
                            data_array=np.concatenate((data_array,newarray),axis=1)
                        else:
                            diff=np.setdiff1d(temp_step,intersect[-1])
                            for i in diff:
                                print('PTRM check at '+str(i)+'K has no corresponding infield measurement, ignoring')
                            newarray=np.array([specarray[temp_step!=diff],sample[temp_step!=diff],site[temp_step!=diff],NRM,PTRM[temp_step!=diff],NRM_x,NRM_y,NRM_z,PTRM_x[temp_step!=diff],PTRM_y[temp_step!=diff],PTRM_z[temp_step!=diff],NRM_sigma,PTRM_sigma[temp_step!=diff],B_lab[temp_step!=diff],steptype[temp_step!=diff],temp_step[temp_step!=diff]])
                            data_array=np.concatenate((data_array,newarray),axis=1)

                        #Add PTRM tail checks
                        ptrm_tail=np.array(ptrm_tail)

                        if len(ptrm_tail)>1:
                            temp_step=ptrm_tail[:,0]
                            sample=np.full(len(ptrm_tail),measurements[measurements.specimen==specimen]['sample'].unique()[0]) #Get sample name
                            site=np.full(len(ptrm_tail),measurements[measurements.specimen==specimen].site.unique()[0]) #Get site name
                            specarray=np.full(len(ptrm_tail),specimen)
                            B_lab=np.full(len(ptrm_tail),field)
                            intersect=data_array[:,(data_array[0]==specimen)&(np.in1d(data_array[-1].astype('float'),temp_step.astype('float')))&(data_array[-2]!='P')]
                            NRM=ptrm_tail[:,3]
                            NRM_sigma=ptrm_tail[:,4]
                            NRM_vector=pmag.dir2cart(np.array([ptrm_tail[:,1],ptrm_tail[:,2],ptrm_tail[:,3]]).T)
                            PTRM_vector=np.array([intersect[8],intersect[9],intersect[10]])
                            PTRM_sigma=intersect[12]
                            PTRM_x=PTRM_vector[0]
                            PTRM_y=PTRM_vector[1]
                            PTRM_z=PTRM_vector[2]
                            NRM_x=NRM_vector[:,0]
                            NRM_y=NRM_vector[:,1]
                            NRM_z=NRM_vector[:,2]
                            PTRM=intersect[4]

                            steptype=np.full(len(ptrm_tail),'T')

                            if len(PTRM)==len(NRM):
                                newarray=np.array([specarray,sample,site,NRM,PTRM,NRM_x,NRM_y,NRM_z,PTRM_x,PTRM_y,PTRM_z,NRM_sigma,PTRM_sigma,B_lab,steptype,temp_step])
                                data_array=np.concatenate((data_array,newarray),axis=1)
                            else:
                                diff=np.setdiff1d(temp_step,intersect[-1])
                                for i in diff:
                                    print('PTRM tail check at '+str(i)+'K has no corresponding zero field measurement, ignoring')
                                newarray=np.array([specarray[temp_step!=diff],sample[temp_step!=diff],site[temp_step!=diff],NRM[temp_step!=diff],PTRM,NRM_x[temp_step!=diff],NRM_y[temp_step!=diff],NRM_z[temp_step!=diff],PTRM_x,PTRM_y,PTRM_z,NRM_sigma[temp_step!=diff],PTRM_sigma,B_lab[temp_step!=diff],steptype[temp_step!=diff],temp_step[temp_step!=diff]])
                                data_array=np.concatenate((data_array,newarray),axis=1)
                    else:
                        print(specimen,'in site',sitename[0],'Not included, not a thellier experiment')
                else:
                    print(specimen,'in site',sitename[0],'Not included, demagnetization not completed')
            except:
                print('Something went wrong with specimen '+specimen+'. Could not convert from MagIC format')
    temps=pd.DataFrame(data_array.T,columns=['specimen','sample','site','NRM','PTRM','NRM_x','NRM_y','NRM_z','PTRM_x','PTRM_y','PTRM_z','NRM_sigma','PTRM_sigma','B_lab','steptype','temp_step'])
    return(temps)

def generate_arai_plot_table(outputname):

    """
    Generates a DataFrame with points on an Araiplot. Inputs: outputname (must be string)
    """
    #This cell constructs the 'measurements' dataframe with samples and sites added
    status,measurements=cb.add_sites_to_meas_table('./')
    measurements=measurements[measurements.specimen.str.contains('#')==False]
    measurements_old=measurements
    measurements=measurements[measurements.experiment.str.contains('LP-PI-TRM')]
    temps=convert_intensity_measurements(measurements)

    temps['correction']=1
    temps['s_tensor']=np.nan
    temps['aniso_type']=np.nan

    spec=pd.read_csv('specimens.txt',skiprows=1,sep='\t')

    #Create the anisotropy tensors if they don't already exist.
    print("Couldn't find Anisotropy Tensors, Generating...")

    #Tensor for ATRM
    ipmag.atrm_magic('measurements.txt')
    try:
        spec_atrm=pd.read_csv('specimens.txt',sep='\t',skiprows=1)
        spec_atrm=spec_atrm.dropna(subset=['method_codes'])
        spec_atrm=spec_atrm[spec_atrm.method_codes.str.contains('LP-AN-TRM')]
        for specimen in spec_atrm.specimen.unique():
            temps.loc[temps.specimen==specimen,'s_tensor']=spec_atrm.loc[spec_atrm.specimen==specimen,'aniso_s'].iloc[0]
            temps.loc[temps.specimen==specimen,'aniso_type']=':DA-AC-ATRM'
    except:
        pass
    #Tensor for AARM
    ipmag.aarm_magic('measurements.txt')
    try:
        spec_aarm=pd.read_csv('specimens.txt',sep='\t',skiprows=1)
        spec_aarm=spec_aarm.dropna(subset=['method_codes'])
        spec_aarm=spec_aarm[spec_aarm.method_codes.str.contains('LP-AN-ARM')]
        for specimen in spec_aarm.specimen.unique():
            temps.loc[temps.specimen==specimen,'s_tensor']=spec_aarm.loc[spec_aarm.specimen==specimen,'aniso_s'].iloc[0]
            temps.loc[temps.specimen==specimen,'aniso_type']=':DA-AC-AARM'
    except:
        pass

    #Get the best fitting hyperbolic tangent for the NLT correction.
    temps['NLT_beta']=np.nan
    NLTcorrs=measurements_old[measurements_old['method_codes']=='LP-TRM:LT-T-I']
    for specimen in NLTcorrs.specimen.unique():
        meas_val=NLTcorrs[NLTcorrs['specimen']==specimen]
        try:
            ab,cov = curve_fit(NLTsolver, meas_val['treat_dc_field'].values*1e6, meas_val['magn_moment'].values/meas_val['magn_moment'].iloc[-1], p0=(max(meas_val['magn_moment']/meas_val['magn_moment'].iloc[-1]),1e-2))
            temps.loc[temps.specimen==specimen,'NLT_beta']=ab[1]
        except RuntimeError:
            print("-W- WARNING: Can't fit tanh function to NLT data for "+specimen)

    #Get the cooling rate correction
    meas_cool=measurements_old[measurements_old.method_codes.str.contains('LP-CR-TRM')].dropna(subset=['description'])

    for specimen in meas_cool.specimen.unique():
        specframe=meas_cool[meas_cool.specimen==specimen]
        vals=specframe.description.str.split(':').values
        crs=np.array([])
        for val in vals:
            crs=np.append(crs,float(val[1]))
        magn_moments=specframe['magn_moment'].values
        avg_moment=np.mean(magn_moments[crs==max(crs)])
        norm_moments=magn_moments/avg_moment
        croven=max(crs)
        crlog=np.log(croven/crs)
        try:
            m,c=np.polyfit(crlog,norm_moments,1)
            sample=specframe['sample'].iloc[0]
            cr_real=samples[samples['sample']==sample].cooling_rate.values/5.256e+11
            cr_reallog=np.log(croven/cr_real)
            cfactor=1/(c+m*cr_reallog)[0]
            temps.loc[temps.specimen==specimen,'correction']=temps.loc[temps.specimen==specimen,'correction']*cfactor
        except TypeError:
            print('Something went wrong with estimating the cooling rate correction for specimen '+specimen+ '. Check that you used the right cooling rate.')

    #Save the dataframe to output.
    temps=temps.dropna(subset=['site'])
    temps.to_csv(outputname+'.csv',index=False)


def maketempsfile(fname):
    """Imports a csv file to this module for use"""
    temps=pd.read_csv(fname)
    temps.set_index([list(range(0,len(temps)))]) #Make sure indexes are unique
    specimens = temps.specimen.unique() #Get specimen list
    return(temps)

def convert(a):
    """Converts data from MagIC format into BiCEP GUI format"""
    convert_button.description='Converting..'
    generate_arai_plot_table('arai_data')
    temps=maketempsfile('arai_data.csv')
    convert_button.description='Convert MagIC data'

def plot_line_base(ax,specimen,min_temp,max_temp,GUI=False):
    """Plots data onto the Arai plot. Does not fit a line to this data"""
    specdf=temps[temps.specimen==specimen]

    IZZI=specdf[(specdf.steptype=='IZ')|(specdf.steptype=='ZI')]
    IZZI_trunc=IZZI[(IZZI.temp_step>=min_temp+273)&(IZZI.temp_step<=max_temp+273)]
    P=specdf[(specdf.steptype=='P')]
    T=specdf[(specdf.steptype=='T')]
    if GUI==True:
        try:
            P_trunc=P[(P.temp_step<=max_temp+273)].iloc[:-1]
            drat=get_drat(IZZI,IZZI_trunc,P_trunc)
            dratbox.description='DRAT: %2.1f'%drat
        except:
            dratbox.description='DRAT: '
    NRM0=specdf.iloc[0].NRM
    PTRMmax=max(specdf.PTRM)/NRM0
    lines=ax.plot(IZZI.PTRM/NRM0,IZZI.NRM/NRM0,'k',linewidth=1)
    emptydots=ax.plot(IZZI.PTRM/NRM0,IZZI.NRM/NRM0,'o',markerfacecolor='None',markeredgecolor='black',label='Not Used')
    ptrm_check=ax.plot(P.PTRM/NRM0,P.NRM/NRM0,'^',markerfacecolor='None',markeredgecolor='black',markersize=10,label='PTRM Check')
    md_check=ax.plot(T.PTRM/NRM0,T.NRM/NRM0,'s',markerfacecolor='None',markeredgecolor='black',markersize=10)
    ax.set_ylim(0,max(IZZI.NRM/NRM0)*1.1)
    ax.set_xlim(0,PTRMmax*1.1)

    IZ=IZZI_trunc[IZZI_trunc.steptype=='IZ']
    ZI=IZZI_trunc[IZZI_trunc.steptype=='ZI']

    iz_plot=ax.plot(IZ.PTRM/NRM0,IZ.NRM/NRM0,'o',markerfacecolor='b',markeredgecolor='black',label='I step')
    zi_plot=ax.plot(ZI.PTRM/NRM0,ZI.NRM/NRM0,'o',markerfacecolor='r',markeredgecolor='black',label='Z step')
    ax.set_ylabel('NRM/NRM$_0$')
    ax.set_xlabel('PTRM/NRM$_0$')
    return(lines,emptydots,ptrm_check,md_check,iz_plot,zi_plot)

def plot_zijd(ax,specimen,min_temp,max_temp):
    """Plots Zijderveld plot of zero field steps for a paleointensity experiment"""
    #Get the NRM data for the specimen
    IZZI=temps.loc[(temps.specimen==specimen)&((temps.steptype=='IZ')|(temps.steptype=='ZI'))]
    IZZI_trunc=IZZI[(temps.temp_step>=min_temp+273)&(temps.temp_step<=max_temp+273)]
    NRM_dirs=IZZI.loc[:,'NRM_x':'NRM_z'].values
    NRM_trunc_dirs=IZZI_trunc.loc[:,'NRM_x':'NRM_z'].values
    try:
        mad=get_mad(IZZI_trunc)
        madbox.description='MAD: %2.1f'%mad
    except:
        madbox.description='MAD: '

    #Plot axis
    ax.axvline(0,color='k',linewidth=1)
    ax.axhline(0,color='k',linewidth=1)

    #Plot NRM directions
    ax.plot(NRM_dirs[:,0],NRM_dirs[:,1],'k')
    ax.plot(NRM_dirs[:,0],NRM_dirs[:,2],'k')

    #Plot NRM directions in currently selected temperature range as closed symbols
    ax.plot(NRM_trunc_dirs[:,0],NRM_trunc_dirs[:,1],'ko')
    ax.plot(NRM_trunc_dirs[:,0],NRM_trunc_dirs[:,2],'rs')

    #Plot open circles for all NRM directions as closed symbols
    ax.plot(NRM_dirs[:,0],NRM_dirs[:,1],'o',markerfacecolor='None',markeredgecolor='k')
    ax.plot(NRM_dirs[:,0],NRM_dirs[:,2],'s',markerfacecolor='None',markeredgecolor='k')

    #Perform PCA fit to data
    if len(IZZI_trunc)>2:
        pca=PCA(n_components=3)
        pca=pca.fit(NRM_trunc_dirs)
        length, vector=pca.explained_variance_[0], pca.components_[0]
        vals=pca.transform(NRM_trunc_dirs)[:,0]
        v = np.outer(vals,vector)

        #Plot PCA line fit
        ax.plot(pca.mean_[0]+v[:,0],pca.mean_[1]+v[:,1],'g')
        ax.plot(pca.mean_[0]+v[:,0], pca.mean_[2]+v[:,2],'g')

        #
        NRM_vect=np.mean(NRM_trunc_dirs,axis=0)
        NRM_mean_magn=np.sqrt(sum(NRM_vect**2))
        vector_magn=np.sqrt(sum(vector**2))
        dang=np.degrees(np.arccos(np.abs(np.dot(NRM_vect,vector)/(NRM_mean_magn*vector_magn))))

        dangbox.description='DANG: %2.1f'%dang
    else:
        dangbox.description='DANG: '

    ax.set_xlabel('x, $Am^2$')
    ax.set_ylabel('y,z, $Am^2$')
    ax.axis('equal')
    ax.relim()

def circleplot(site,fit,i,ax,temperatures,legend=False,linewidth=2,title=None,tangent=False):
    """Plots Circle fits sampled from the posterior distribution
    (using the BiCEP method) to the Arai plot data. Plots tangent
    to the circle as a slope if tangent=True"""

    #Get information on maximum pTRM for rescaling of circle
    specimenlist=temps[temps.site==site].specimen.unique()
    specimen=specimenlist[i]
    specdf=temps[temps.specimen==specimen]
    IZZI=specdf[(specdf.steptype=='IZ')|(specdf.steptype=='ZI')]
    NRM0=specdf.NRM.iloc[0]
    PTRMmax=max(IZZI.PTRM)/NRM0
    if temperatures!=None:
        IZZI_trunc=IZZI[(IZZI.temp_step>=temperatures[specimen][0,0])&(IZZI.temp_step<=temperatures[specimen][0,1])]
    else:
        IZZI_trunc=IZZI
    minNRM=min(IZZI_trunc.NRM/NRM0)
    minPTRM=min(IZZI_trunc.PTRM/NRM0)


    #Parameters for the circle fit
    c=np.random.choice(range(len(fit['R'][:,i])),100)
    thetas=np.linspace(0,2*np.pi,1000)
    NRM0=temps[temps.specimen==specimen].iloc[0].NRM

    #Circle x and y values for circle plot.
    xs=(fit['x_c'][c,i][:,np.newaxis])*PTRMmax+minPTRM+fit['R'][c,i][:,np.newaxis]*np.cos(thetas)*PTRMmax
    ys=fit['y_c'][c,i][:,np.newaxis]+minNRM+fit['R'][c,i][:,np.newaxis]*np.sin(thetas)

    #Plot Circles
    ax.plot(xs.T,ys.T,'-',color='lightgreen',alpha=0.2,linewidth=linewidth,zorder=-1);
    ax.plot(100,100,'-',color='lightgreen',label='Circle Fits');

    #Find tangents to the circle:
    if tangent==True:
        slope_ideal=-1/np.tan(np.median(fit['phi'][:,i]))/PTRMmax
        x_i=np.median(fit['dist_to_edge'][:,i])*np.cos(np.median(fit['phi'][:,i]))*PTRMmax+minPTRM
        y_i=np.median(fit['dist_to_edge'][:,i])*np.sin(np.median(fit['phi'][:,i]))+minNRM

        ax.plot(x_i,y_i,'ko')
        c=y_i-slope_ideal*x_i
        d=-c/slope_ideal
        ax.plot([0,d],[c,0],'k',linestyle='--')

    #Add legend and title to plot
    if legend==True:
        ax.legend(fontsize=10);
    if title!=None:
        ax.set_title(title,fontsize=20,loc='left')

def regplot(fit,ax,specimenlist,legend=False,title=None):
    """Plots B vs k for all specimens in a site given a BiCEP or unpooled fit"""
    B_lab_list=[]
    for specimen in specimenlist:
        B_lab_list.append(temps[temps.specimen==specimen].B_lab.unique()*1e6)
    try:
        Bs=fit['int_real']
        mink,maxk=np.amin(fit['k']),np.amax(fit['k'])
        minB,maxB=fit['c']*mink+fit['int_site'],fit['c']*maxk+fit['int_site']
        c=np.random.choice(range(len(minB)),100)
        ax.plot([mink,maxk],[minB[c],maxB[c]],color='skyblue',alpha=0.12)
    except:
        Bs=fit['slope']*np.array(B_lab_list).T
    ax.set_xlabel(r'$\vec{k}$');
    ax.plot(np.percentile(fit['k'],(2.5,97.5),axis=0),[np.median(Bs,axis=0),np.median(Bs,axis=0)],'k')
    ax.plot([np.median(fit['k'],axis=0),np.median(fit['k'],axis=0)],np.percentile(Bs,(2.5,97.5),axis=0),'k')
    ax.plot(np.median(fit['k'],axis=0),np.median(Bs,axis=0),'o',markerfacecolor='lightgreen',markeredgecolor='k')
    ax.axvline(0,color='k',linewidth=1)
    if title!=None:
        ax.set_title(title,fontsize=20,loc='left')

def display_gui():
    """Displays the specimen plots for BiCEP GUI"""
    for axis in ax:
        axis.cla()
    plot_line_base(ax[0],specimen_wid.value,lower_temp_wid.value,upper_temp_wid.value,GUI=True) #Base Arai plot
    plot_zijd(ax[1],specimen_wid.value,lower_temp_wid.value,upper_temp_wid.value) #Zijderveld plot

    try:
        fit=fits[site_wid.value]
        specimenlist=np.array(specimen_wid.options)
        specimen=specimen_wid.value
        i=np.where(np.array(specimenlist)==specimen)[0][0]
        circleplot(site_wid.value,fit,i,ax[0],ktemp)
    except:
        pass
    fig.set_tight_layout(True)
    fig_2.set_tight_layout(True)

def plot_site_plot(fit):
    """Plots the plot of k vs B_anc and the histogram for the site fits on BiCEP GUI"""
    ax_2[0].axhline(np.median(fit['int_site']),color='k')
    ax_2[1].axhline(np.median(fit['int_site']),color='k')
    ax_2[1].hist(fit['int_site'],color='skyblue',bins=100,density=True,orientation='horizontal')
    specimenlist=specimen_wid.options
    regplot(fit,ax_2[0],specimenlist)
    ax_2[0].yaxis.tick_right()
    ax_2[1].yaxis.tick_right()
    ax_2[1].yaxis.set_label_position('right')
    ax_2[0].set_ylim(min(np.percentile(fit['int_real'],2.5,axis=0))*0.9,max(np.percentile(fit['int_real'],97.5,axis=0))*1.1)
    ax_2[0].set_xlim(min(min(np.percentile(fit['k'],2.5,axis=0))*1.1,min(np.percentile(fit['k'],2.5,axis=0))*0.9),max(max(np.percentile(fit['k'],97.5,axis=0))*1.1,max(np.percentile(fit['k'],97.5,axis=0))*0.9))
    currspec=specimen_wid.value
    specindex=np.where(specimenlist==currspec)
    specindex=specindex[0]
    ax_2[0].plot(np.median(fit['k'][:,specindex]),np.median(fit['int_real'][:,specindex]),'o',markeredgecolor='r',markerfacecolor='r')
    ax_2[1].set_ylabel('$B_{anc}$')
    ax_2[1].set_xlabel('Probability Density')


def display_site_plot():
    """Updates everything needed once the BiCEP method has been applied to a site"""
    try:
        fit=fits[site_wid.value]
        ax_2[0].cla()
        ax_2[1].cla()
        plot_site_plot(fit)
        display_sampler_diags(fit)
        display_specimen_ring()
    except:
        #If no fit for this site yet, delete everything and make a new plot!
        ax_2[0].cla()
        ax_2[1].cla()
        rhatlabel.description='R_hat:'
        nefflabel.description='n_eff:'
        banclabel.description='B_anc:'

def display_specimen_ring():
    """Displays a red circle around the currently selected
    specimen in the site plot of BiCEP GUI"""
    try:
        fit=fits[site_wid.value]
        #Maybe not the most efficent way of doing things,
        #need to loop through matplotlib elements to find
        #any red circles that already exist

        for line in ax_2[0].lines:
            if line.properties()['markeredgecolor']=='r':
                line.remove()
        specimenlist=specimen_wid.options
        currspec=specimen_wid.value
        specindex=np.where(np.array(specimenlist)==currspec)
        ax_2[0].plot(np.median(fit['k'][:,specindex]),np.median(fit['int_real'][:,specindex]),'o',markeredgecolor='r',markerfacecolor='None')
        circleplot(site_wid.value,fit,specindex,ax[0],temperatures)

    except:
        pass


def on_change(change):
    """Update GUI on changing one of our site, specimen, temperature dropdowns"""
    ax[0].cla()
    #If we're changing the site dropdown, we need to replot the site plots and change the specimen options
    if (change.owner==site_wid):
        specimen_wid.options=temps[temps.site==site_wid.value].specimen.unique()
        display_site_plot()
    #If we're changing the site or specimen dropdown, we need to update the temperature steps.
    if (change.owner==site_wid)|(change.owner==specimen_wid):
        lower_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273
        upper_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273
        lower_temp_wid.value=temperatures[specimen_wid.value][0,0]
        upper_temp_wid.value=temperatures[specimen_wid.value][0,1]

    #If we're changing the specimen plot, we display a red circle around the currently selected specimen on the site plot
    if (change.owner==specimen_wid):
        display_specimen_ring()

    #We always need to redraw the specimen dropdown.
    display_gui()


def on_button_clicked(a):
    """GUI function for saving specimen min and max temperatures (saves to file)"""
    temperatures[specimen_wid.value]=np.array([[lower_temp_wid.value,upper_temp_wid.value]])
    with open('specimen-temperatures.pickle', 'wb') as tempdict:
        pickle.dump(temperatures, tempdict)

def get_sampler_diags(fit):
    """Returns useful sampler diagnostics for a particular MCMC fit with pystan"""
    rhat=fit.summary()['summary'][:,-1]
    rhat_worst=rhat[np.abs(1-rhat)==max(np.abs(1-rhat))][0]
    n_eff_int_site=int(fit.summary()['summary'][0,-2])
    return rhat_worst,n_eff_int_site

def display_sampler_diags(fit):
    """Displays the worst R_hat and n_eff for B_anc for the fit for the MCMC fit for a site in BiCEP_GUI"""
    rhat_worst,n_eff_int_site=get_sampler_diags(fit)
    if (rhat_worst>1.1)|(rhat_worst<0.9):
        rhatlabel.button_style='danger'
    else:
        rhatlabel.button_style='success'
    if n_eff_int_site<1000:
        nefflabel.button_style='warning'
    else:
        nefflabel.button_style='success'

    rhatlabel.description='R_hat: %1.2f'%rhat_worst
    nefflabel.description='n_eff:'+str(n_eff_int_site)
    minB,maxB=np.percentile(fit['int_site'],(2.5,97.5),axis=0)
    banclabel.description='B_anc %3.1f'%minB+'- %3.1f'%maxB+' μT'

def get_site_dist(a):
    """Runs the MCMC sampler and updates the GUI"""
    process_wid.description='Processing..'
    for ax in ax_2:
        ax.cla()
    specimenlist=np.array(specimen_wid.options)
    methods={'Slow, more accurate':model_circle_slow,'Fast, less accurate':model_circle_fast}
    for key in temperatures:
        ktemp[key]=temperatures[key]+273
    fit,newmethcodes,newcolumns=BiCEP_fit(specimenlist,ktemp,model=methods[method_wid.value],priorstd=5,n_samples=n_samples_wid.value)
    plot_site_plot(fit)
    display_sampler_diags(fit)
    fits[site_wid.value]=fit
    for specimen in newmethcodes.keys():
        spec_method_codes[specimen]+=newmethcodes[specimen]
        spec_extra_columns[specimen]=newcolumns[specimen]
    display_specimen_ring()
    display_gui()
    process_wid.description='Process Site Data'

def newfile(a):
    """Sets up the GUI with a new file converted from arai data"""
    global been_pressed
    if been_pressed==False:
        global temps,site_wid,specimen_wid,lower_temp_wid,upper_temp_wid,save_wid,temperatures,fig,ax,spec_method_codes,site_method_codes
        been_pressed=True
        temps=pd.read_csv('arai_data.csv')
        spec_method_codes={specimen:'IE-BICEP' for specimen in temps.specimen.unique()}
        site_method_codes={site:'IE-BICEP' for site in temps.site.unique()}
        try:
            with open("specimen-temperatures.pickle",'rb') as tempdict:
                temperatures=pickle.load(tempdict)
            if len(np.intersect1d(tempdict.keys(),temps.specimen.unique()))==0:
                temperatures={specimen:np.array([[temps[temps.specimen==specimen].temp_step.unique()[0]-273,temps[temps.specimen==specimen].temp_step.unique()[-1]-273]]) for specimen in temps.specimen.unique()}
            else:
                pass
        except:
            temperatures={specimen:np.array([[temps[temps.specimen==specimen].temp_step.unique()[0]-273,temps[temps.specimen==specimen].temp_step.unique()[-1]-273]]) for specimen in temps.specimen.unique()}

        site_wid.options=temps.site.unique()

        specimen_wid.options=temps[temps.site==site_wid.value].specimen.unique()

        lower_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273

        upper_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273
        site_wid.observe(on_change)
        specimen_wid.observe(on_change)
        lower_temp_wid.observe(on_change)
        upper_temp_wid.observe(on_change)
        save_wid.on_click(on_button_clicked)
        display_gui()
    else:
        pass

def examplefile(a):
    """Sets up the GUI with the example dataset of Cych et al (in prep)"""
    global been_pressed
    if been_pressed==False:
        global temps,site_wid,specimen_wid,lower_temp_wid,upper_temp_wid,save_wid,temperatures,fig,ax,spec_method_codes,site_method_codes
        temps=pd.read_csv('arai_data_example.csv')
        been_pressed=True
        spec_method_codes={specimen:'IE-BICEP' for specimen in temps.specimen.unique()}
        site_method_codes={site:'IE-BICEP' for site in temps.site.unique()}
        try:
            with open("specimen-temperatures.pickle",'rb') as tempdict:
                temperatures=pickle.load(tempdict)
            if len(np.intersect1d(tempdict.keys(),temps.specimen.unique()))==0:
                temperatures={specimen:np.array([[temps[temps.specimen==specimen].temp_step.unique()[0]-273,temps[temps.specimen==specimen].temp_step.unique()[-1]-273]]) for specimen in temps.specimen.unique()}
            else:
                pass
        except:
            temperatures={specimen:np.array([[temps[temps.specimen==specimen].temp_step.unique()[0]-273,temps[temps.specimen==specimen].temp_step.unique()[-1]-273]]) for specimen in temps.specimen.unique()}

        site_wid.options=temps.site.unique()

        specimen_wid.options=temps[temps.site==site_wid.value].specimen.unique()

        lower_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273

        upper_temp_wid.options=temps[temps.specimen==specimen_wid.value].temp_step.unique()-273

        site_wid.observe(on_change)
        specimen_wid.observe(on_change)
        lower_temp_wid.observe(on_change)
        upper_temp_wid.observe(on_change)
        save_wid.on_click(on_button_clicked)
        display_gui()
    else:
        pass


row_layout_buttons = widgets.Layout(
    width='60%',
    display='flex',
    flex_flow='row',
    justify_content='space-around',
    margin='1000px left'
)
row_layout = widgets.Layout(
    width='100%',
    display='flex',
    flex_flow='row',
    justify_content='space-around'
)

def save_magic_tables(a):
    """Saves data from the currently displayed site to the GUI"""
    fit=fits[site_wid.value]
    sitestable=pd.read_csv('sites.txt',skiprows=1,sep='\t')
    sitestable.loc[sitestable.site==site_wid.value,'int_abs_min']=round(np.percentile(fit['int_site'],2.5),1)/1e6
    sitestable.loc[sitestable.site==site_wid.value,'int_abs_max']=round(np.percentile(fit['int_site'],97.5),1)/1e6
    sitestable.loc[sitestable.site==site_wid.value,'int_abs']=round(np.percentile(fit['int_site'],50),1)/1e6
    specimenstable=pd.read_csv('specimens.txt',skiprows=1,sep='\t')
    speclist=specimen_wid.options
    for i in range(len(speclist)):

        specimen=speclist[i]
        specfilter=(~specimenstable.method_codes.str.contains('LP-AN').fillna(False))&(specimenstable.specimen==specimen)
        specimenstable.loc[specfilter,'int_abs_min']=round(np.percentile(fit['int_real'][:,i],2.5),1)/1e6
        specimenstable.loc[specfilter,'int_abs_max']=round(np.percentile(fit['int_real'][:,i],97.5),1)/1e6
        specimenstable.loc[specfilter,'int_abs']=round(np.percentile(fit['int_real'][:,i],50),1)/1e6
        specimenstable.loc[specfilter,'int_k_min']=round(np.percentile(fit['k'][:,i],2.5),3)
        specimenstable.loc[specfilter,'int_k_max']=round(np.percentile(fit['k'][:,i],97.5),3)
        specimenstable.loc[specfilter,'int_k']=round(np.percentile(fit['k'][:,i],50),3)
        specimenstable.loc[specfilter,'meas_step_min']=ktemp[specimen][0,0]
        specimenstable.loc[specfilter,'meas_step_max']=ktemp[specimen][0,1]
        method_codes=spec_method_codes[specimen].split(':')
        method_codes=list(set(method_codes))
        newstr=''
        for code in method_codes[:-1]:
            newstr+=code
            newstr+=':'
        newstr+=method_codes[-1]
        specimenstable.loc[specfilter,'method_codes']=spec_method_codes[specimen]

        extra_columns=spec_extra_columns[specimen]
        for col in extra_columns.keys():
            specimenstable.loc[specfilter,col]=extra_columns[col]
    sitestable.loc[sitestable.site==site_wid.value,'method_codes']=site_method_codes[site_wid.value]
    specimenstable['meas_step_unit']='Kelvin'
    sitestable=sitestable.fillna('')
    specimenstable=specimenstable.fillna('')
    sitesdict=sitestable.to_dict('records')
    specimensdict=specimenstable.to_dict('records')
    pmag.magic_write('sites.txt',sitesdict,'sites')
    pmag.magic_write('specimens.txt',specimensdict,'specimens')

def save_figures(a):
    """Saves figures from GUI depending on widgets"""
    objdict={'Specimen Plot':fig,'Site Plot':fig_2}
    value={'Specimen Plot':specimen_wid.value,'Site Plot':site_wid.value}
    objdict[figchoice.value].savefig(value[figchoice.value]+'_BiCEP_fit.'+figformats.value)







#Objects global to all the functions in this module
fits={} #Fits for various sites are stored here- this can use a lot of memory!
ktemp={} #Temperatures (In Kelvin) used for a specimen at the time a site fit has been performed. Only done once site fit is performed.
spec_extra_columns={} #Additional columns for the MagIC tables, e.g. if corrections were applied.



#GUI widgets- these widgets are what are used in the BiCEP_GUI notebook/voila page, which allows you to display them.
#Their interaction is linked to the functions in this module.


#File picker widget- probably not necessary as sites, specimens etc already there.
convert_button = widgets.Button(description='Convert MagIC Data')

convert_button.on_click(convert)


#Been_pressed flag stops you from loading a new dataset when one is already loaded (this breaks the GUI because the dropdown boxes try and update their options before they know what those options are).
global been_pressed
been_pressed=False


#Specimen/Interpretation Selection, Arai plot etc.
site_wid= widgets.Dropdown(
    description='Site:')

specimen_wid= widgets.Dropdown(
    options=[],
    description='Specimen:')

lower_temp_wid=widgets.Dropdown(
    options=[],
    description='Temperatures (Low):',
    style={"description_width":"initial"}
    )

upper_temp_wid=widgets.Dropdown(
    options=[],
    description='(High):',
    style={"description_width":"initial"})

save_wid=widgets.Button(description='Save Temperatures')


newfile_wid=widgets.Button(description='Use New File',
                          style={"description_width":"initial"})
examplefile_wid=widgets.Button(description='Use Example File',
                              style={"description_width":"initial"})
figsave=widgets.Button(description='Save Figures')
figchoice=widgets.Dropdown(options=['Specimen Plot','Site Plot'])
figformats=widgets.Dropdown(description='Format:',options=['pdf','png','jpg','svg','tiff'])
figsave.on_click(save_figures)
newfile_wid.on_click(newfile)
examplefile_wid.on_click(examplefile)

madbox=widgets.Button(description='MAD:',disabled=True)
dangbox=widgets.Button(description='DANG:',disabled=True)
dratbox=widgets.Button(description='DRAT:',disabled=True)
filebox=widgets.HBox([newfile_wid,examplefile_wid],grid_area="filebox")
tempbox=widgets.VBox([lower_temp_wid,upper_temp_wid],grid_area="tempbox")
specbox=widgets.VBox([site_wid,specimen_wid],grid_area="specbox")

savebox=widgets.HBox([save_wid,figsave,figchoice,figformats])
dirbox=widgets.HBox([madbox,dangbox])
critbox=widgets.VBox([dirbox,dratbox])
specplots=widgets.Output(grid_area="specplots")
dropdowns=widgets.HBox([specbox,tempbox,critbox],grid_area="dropdowns")

#fullbox gives the entire specimen processing box
fullbox=widgets.Box(children=[filebox,dropdowns,specplots,savebox],title='Specimen Processing',
        layout=widgets.Layout(
            width='100%',
            flex_flow='column',
            align_content='space-around',
            align_items='flex-start')
       )
#Make a plot in the specplots box
with specplots:
    fig,ax=plt.subplots(1,2,figsize=(9,3))
    fig.canvas.header_visible = False
    plt.tight_layout()
dangbox.button_style='info'
madbox.button_style='info'
dratbox.button_style='info'

#GUI widgets for the site processing box
n_samples_wid=widgets.IntSlider(min=3000,max=100000,value=30000,step=1000,description='n samples')
method_wid=widgets.Dropdown(options=['Slow, more accurate','Fast, less accurate'],description='Sampler:')
process_wid=widgets.Button(description='Process Site Data')
process_wid.on_click(get_site_dist)
rhatlabel=widgets.Button(descriptixon='R_hat:',disabled=True)
nefflabel=widgets.Button(description='n_eff:',disabled=True)
banclabel=widgets.Button(description='B_anc:',disabled=True,display='flex',flex_flow='column',align_items='stretch',layout=widgets.Layout(width='auto', height=rhatlabel.layout.height))
sampler_diag=widgets.HBox([rhatlabel,nefflabel])
sampler_buttons=widgets.VBox([sampler_diag,banclabel])
sampler_pars=widgets.VBox([n_samples_wid,method_wid])
sampler_line=widgets.HBox([sampler_pars,sampler_buttons])
banclabel.button_style='info'
nefflabel.button_style='info'
rhatlabel.button_style='info'
siteplots=widgets.Output()
with siteplots:
    fig_2,ax_2=plt.subplots(1,2,figsize=(6.4,3),sharey=True)
    fig_2.canvas.header_visible = False



savetables=widgets.Button(description='Save to MagIC tables')
savetables.on_click(save_magic_tables)
sitesave=widgets.HBox([savetables])

fullbox2=widgets.VBox([process_wid,sampler_line,siteplots,sitesave],title='Site Processing')
specpage=widgets.Accordion([fullbox])
sitepage=widgets.Accordion([fullbox2])
specpage.set_title(0,'Specimen Processing')
sitepage.set_title(0,'Site Processing')
gui=widgets.VBox([specpage,sitepage])
