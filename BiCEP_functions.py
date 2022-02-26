import pandas as pd
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
import sys
import arviz as az

model_circle_fast=pickle.load(open('model_circle_fast.pkl','rb'))
model_circle_slow=pickle.load(open('model_circle_slow.pkl','rb'))
def sufficient_statistics(ptrm, nrm):
    """
    inputs list of ptrm and nrm data and computes sufficent statistcs needed
    for computations

    Inputs
    ------
    ptrm: list
    list of ptrm data

    nrm: list
    list of nrm data

    Returns
    -------
    dict containing mean ptrm and nrm, and covariances in xx, xy and yy.
    """

    corr = np.cov( np.stack((ptrm, nrm), axis=0) )

    return {'xbar': np.mean(ptrm), 'ybar': np.mean(nrm), 'S2xx': corr[0,0], 'S2yy': corr[1,1], 'S2xy': corr[0,1] }

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

def bestfit_line(ptrm, nrm):
    """
    Returns the slope and intercept of the best fit line to a set of
    pTRM and NRM data using York Regression

    Inputs
    ------
    ptrm: list or array
    list of pTRM data

    nrm: list or array
    list of NRM data

    Returns
    -------
    dictionary of slope and intercept for best fitting line.
    """
    stat = sufficient_statistics(ptrm, nrm)

    w = .5*(stat['S2xx'] - stat['S2yy'])/stat['S2xy']
    m = -w-np.sqrt(w**2+1)
    b = stat['ybar']-m*stat['xbar']
    return {'slope': m, 'intercept': b }

def get_drat(IZZI,IZZI_trunc,P):
    """Calculates the difference ratio (DRAT) of pTRM checks
    (Selkin and Tauxe, 2000) to check for alteration

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of all in field and
    zero field measurements for a specimen.

    IZZI_trunc: pandas.DataFrame
    DataFrame object- same as IZZI but truncated only for temperatures
    used in interpretation

    P: pandas.DataFrame
    DataFrame object containing pTRM checks up to the
    maximum temperature of an interpretation

    Returns
    -------
    absdiff: float
    maximum DRAT for all pTRM check measurements.
    returns zero if the interpretation is not valid.
    """
    try:
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
    except:
        return 0

def get_mad(IZZI,pca):
    """
    Calculates the free Maximum Angle of Deviation (MAD) of Kirshvink et al (1980)

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    pca: scikitlearn.decomposition.PCA object
    pca used to fit the vector direction.

    Returns
    -------
    mad: float
    maximum angle of deviation for that intepretation
    """
    try:
        fit=pca.fit(IZZI.loc[:,'NRM_x':'NRM_z'].values).explained_variance_
        return np.degrees(np.arctan(np.sqrt((fit[2]+fit[1])/(fit[0]))))
    except:
        return 0
def get_dang(NRM_trunc_dirs,pca):
    """
    Calculates the Deviation Angle
    Inputs
    ------
    NRM_trunc_dirs: numpy.ndarray
    Vector directions for zero field measurements for specimen

    pca: scikitlearn.decomposition.PCA object
    pca used to fit the vector direction.

    Returns
    -------
    dang: float
    deviation angle for that intepretation
    """
    try:
        length, vector=pca.explained_variance_[0], pca.components_[0]
        NRM_vect=np.mean(NRM_trunc_dirs,axis=0)
        NRM_mean_magn=np.sqrt(sum(NRM_vect**2))
        vector_magn=np.sqrt(sum(vector**2))
        return(np.degrees(np.arccos(np.abs(np.dot(NRM_vect,vector)/(NRM_mean_magn*vector_magn)))))
    except:
        return(0)


def calculate_anisotropy_correction(IZZI):
    """
    Calculates anisotropy correction factor for a
    paleointensity interpretation, given an s tensor

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    Returns
    -------
    c: float
    Anisotropy correction factor
    """

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
    """
    Calculates the correction for non linear TRM for a paleointensity interpretation,
    given the anisotropy and cooling rate corrections

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    c: float
    Combined Anisotropy and Cooling Rate correction-
    needed because the nonlinearity is applied after this.

    Returns
    -------
    c: float
    NLT correction factor
    """
    a=np.sum((IZZI.PTRM-np.mean(IZZI.PTRM))*(IZZI.NRM-np.mean(IZZI.NRM)))
    b=a/np.abs(a)*np.sqrt(np.sum((IZZI.NRM-np.mean(IZZI.NRM))**2)/np.sum((IZZI.PTRM-np.mean(IZZI.PTRM))**2))
    beta=IZZI['NLT_beta'].iloc[0]
    correction=c*IZZI.correction.iloc[0]
    B_lab=IZZI.B_lab.iloc[0]*1e6
    total_correction=(np.arctanh(correction*np.abs(b)*np.tanh(beta*B_lab)))/(np.abs(b)*beta*B_lab)
    return(total_correction)

def extract_values(fit,var):
    """
    Extracts the data for variable in the BiCEP fit to a 1d numpy array

    Inputs
    ------
    fit arviz InferenceData:
    site level BiCEP fit object
    
    var str:
    name of variable in fit

    Returns
    -------
    values: array
    1d array of values for that variable in the fit
    """
    values=fit.posterior[var].stack(sample=["chain", "draw"]).values
    return(values)


class ThellierData():
    """
    Class which supports several methods using the BiCEP method in pandas.
    Inputs
    ------
    datafile: string for file name in BiCEP format
    """

    def __init__(self,datafile):
        self.data=pd.read_csv(datafile)
        self.groupType='site'
        try:
            self.redo=pd.read_csv('thellier_gui.redo',delim_whitespace=True,header=None)
        except:
            self.redo=None
        
        self.collections={siteName:SpecimenCollection(self,siteName,self.groupType) for siteName in self.data[self.groupType].unique()}
        

    def __repr__(self):
        reprstr='Set of Thellier Data Containing the '+'S'+self.groupType[1:]+'s:\n'
        for key in self.collections.keys():
            reprstr+= key+'\t('+str(len(self.collections[key].specimens))+' specimens)\n'
        return(reprstr)
    def __getitem__(self,item):
        return(self.collections[item])

    def switch_grouping(self):
        if self.groupType=='site':
            self.groupType='sample'
        else:
            self.groupType='site'
        self.collections={siteName:SpecimenCollection(self,siteName,self.groupType) for siteName in self.data[self.groupType].unique()}



class SpecimenCollection():
    """
    Collection of specimens (site or sample, from ThellierData Dataset)

    Parameters
    ----------

    parentData: ThellierData object
    Set of Thellier Data the site/sample is derived from

    collectionName: string
    Name of specimen/site

    key: string
    Either 'site' for site or 'sample' for sample
    """

    def __init__(self,parentData,collectionName,key):
        self.name=collectionName
        self.key=key
        self.parentData=parentData
        self.data=parentData.data[parentData.data[self.key]==collectionName]
        self.specimens={specimenName:Specimen(self,specimenName) for specimenName in self.data.specimen.unique()}
        try:
            self.fit=az.from_netcdf(self.name+'.nc')
        except:
            self.fit=None
        self.methcodes='IE-BICEP'
        self.artist=None


    def __repr__(self):
        reprstr='Site containing the specimens:\n'
        for specimenName in self.specimens.keys():
            reprstr+=specimenName+'\n'
        return(reprstr)

    def __getitem__(self,item):
        return(self.specimens[item])


    def BiCEP_fit(self,n_samples=30000,priorstd=5,model=None,**kwargs):
        """
        Performs the fitting routine using the BiCEP method for a given list of specimens from a single site.
        Inputs
        ------
        Specimenlist: iterable of specimen names (strings)

        """
        minPTRMs=[]
        minNRMs=[]
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
        i=0
        #try:
        for specimen in self.specimens.values():
            if specimen.active==True:
                specimen.save_changes()
                minPTRM,minNRM,PTRMmax,k,phi,dist_to_edge,sigma,PTRMS,NRMS=specimen.BiCEP_prep()
                NRM0=specimen.NRM0
                minPTRMs.append(minPTRM)
                minNRMs.append(minNRM)
                line=bestfit_line(specimen.IZZI_trunc.PTRM,specimen.IZZI_trunc.NRM)
                B_anc=-line['slope']*specimen.B_lab*specimen.IZZI_trunc.correction.iloc[0]
                B_ancs.append(B_anc)
                Pi,Pj=np.meshgrid(PTRMS,PTRMS)
                Ni,Nj=np.meshgrid(NRMS,NRMS)
                dmax=np.amax(np.sqrt((Pi-Pj)**2+(Ni-Nj)**2))
                centroid=np.sqrt(np.mean(PTRMS)**2+np.mean(NRMS)**2)
                B_lab_list.append(specimen.B_lab)
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
                i+=1

        if model==None:
            if i<7:
                model_circle=model_circle_slow
            else:
                model_circle=model_circle_fast
        else:
            model_circle=model
        fit_circle=model_circle.sampling (
            data={'I':len(pTRMsList),'M':len(lengths),'PTRM':pTRMsList,'NRM':NRMsList,'N':lengths,'PTRMmax':PTRMmaxlist,'B_labs':B_lab_list,'dmax':np.sqrt(dmaxlist),'centroid':centroidlist,'priorstd':priorstd},iter=n_samples,warmup=int(n_samples/2),
            init=[{'k_scale':np.array(klist)*np.array(dist_to_edgelist),'phi':philist,'dist_to_edge':dist_to_edgelist,'int_real':B_ancs}]*4,**kwargs)
        self.fit=az.from_pystan(fit_circle)
        #except:
            #print('Something went wrong trying to do the BiCEP fit, try changing your temperature range')



    def save_magic_tables(self):
        """
        Saves data from the currently displayed site to the GUI

        Inputs
        ------
        None

        Returns
        -------
        None
        """
        fit=self.fit
        sitestable=pd.read_csv(self.key+'s.txt',skiprows=1,sep='\t')
        sitestable.loc[sitestable[self.key]==self.name,'int_abs_min']=round(np.percentile(extract_values(fit,'int_site'),2.5),1)/1e6
        sitestable.loc[sitestable[self.key]==self.name,'int_abs_max']=round(np.percentile(extract_values(fit,'int_site'),97.5),1)/1e6
        sitestable.loc[sitestable[self.key]==self.name,'int_abs']=round(np.percentile(extract_values(fit,'int_site'),50),1)/1e6
        specimenstable=pd.read_csv('specimens.txt',skiprows=1,sep='\t')
        speclist=[spec for spec in self.specimens.keys() if self.specimens[spec].active==True]
        for i in range(len(speclist)):

            specimen=speclist[i]
            specfilter=(~specimenstable.method_codes.str.contains('LP-AN').fillna(False))&(specimenstable.specimen==specimen)
            specimenstable.loc[specfilter,'int_abs_min']=round(np.percentile(extract_values(fit,'int_real')[i],2.5),1)/1e6
            specimenstable.loc[specfilter,'int_abs_max']=round(np.percentile(extract_values(fit,'int_real')[i],97.5),1)/1e6
            specimenstable.loc[specfilter,'int_abs']=round(np.percentile(extract_values(fit,'int_real')[i],50),1)/1e6
            specimenstable.loc[specfilter,'int_k_min']=round(np.percentile(extract_values(fit,'k')[i],2.5),3)
            specimenstable.loc[specfilter,'int_k_max']=round(np.percentile(extract_values(fit,'k')[i],97.5),3)
            specimenstable.loc[specfilter,'int_k']=round(np.percentile(extract_values(fit,'k')[i],50),3)
            specimenstable.loc[specfilter,'meas_step_min']=self[specimen].savedLowerTemp
            specimenstable.loc[specfilter,'meas_step_max']=self[specimen].savedUpperTemp
            method_codes=self[specimen].methcodes.split(':')
            method_codes=list(set(method_codes))
            newstr=''
            for code in method_codes[:-1]:
                newstr+=code
                newstr+=':'
            newstr+=method_codes[-1]
            specimenstable.loc[specfilter,'method_codes']=self[specimen].methcodes

            extra_columns=self[specimen].extracolumnsdict
            for col in extra_columns.keys():
                specimenstable.loc[specfilter,col]=extra_columns[col]
        sitestable.loc[sitestable.site==self.name,'method_codes']=self.methcodes
        specimenstable['meas_step_unit']='Kelvin'
        sitestable=sitestable.fillna('')
        specimenstable=specimenstable.fillna('')
        sitesdict=sitestable.to_dict('records')
        specimensdict=specimenstable.to_dict('records')
        pmag.magic_write('sites.txt',sitesdict,'sites')
        pmag.magic_write('specimens.txt',specimensdict,'specimens')

    def regplot(self,ax,legend=False,title=None):
        """
        Plots B vs k for all specimens in a site given a BiCEP or unpooled fit

        Inputs
        ------
        ax: matplotlib axis
        axes to plot to

        legend: bool
        If set to True, plots a legend

        title: str
        Title for plot. Does not plot title if set to None.
        """
        B_lab_list=[]
        for specimen in self.specimens.values():
            B_lab_list.append(specimen.B_lab)
        try:
            Bs=extract_values(self.fit,'int_real').T
            ks=extract_values(self.fit,'k').T
            int_sites=extract_values(self.fit,'int_site')
            cs=extract_values(self.fit,'c')
            mink,maxk=np.amin(ks),np.amax(ks)
            minB,maxB=cs*mink+int_sites,cs*maxk+int_sites
            c=np.random.choice(range(len(minB)),100)
            ax.plot([mink,maxk],[minB[c],maxB[c]],color='skyblue',alpha=0.12)
        except:
            Bs=extract_values(self.fit,'slope').T*np.array(B_lab_list).T
            ks=extract_values(self.fit,'k').T
        ax.set_xlabel(r'$\vec{k}$');
        ax.plot(np.percentile(ks,(2.5,97.5),axis=0),[np.median(Bs,axis=0),np.median(Bs,axis=0)],'k')
        ax.plot([np.median(ks,axis=0),np.median(ks,axis=0)],np.percentile(Bs,(2.5,97.5),axis=0),'k')
        ax.plot(np.median(ks,axis=0),np.median(Bs,axis=0),'o',markerfacecolor='lightgreen',markeredgecolor='k')
        ax.axvline(0,color='k',linewidth=1)
        if title!=None:
            ax.set_title(title,fontsize=20,loc='left')

    def get_specimen_rhats(self):
        """
        Finds the worst Rhat va; for each specimen and assigns it to that specimen
        """
        rhats_orig=az.rhat(self.fit)
        rhats=rhats_orig.to_stacked_array("rhats",sample_dims=[]).values
        worst_rhat=rhats[(1-rhats)**2==max((1-rhats)**2)][0]
        specrhatsarray=[]
        specrhats=rhats_orig.drop_vars(['int_site','sd_site','c']).data_vars
        for i in specrhats:
            specrhatsarray.append(specrhats[i].values)
        specrhatsarray=np.array(specrhatsarray)
        speclist=[spec for spec in self.specimens.keys() if self.specimens[spec].active==True]
        for j in range(len(speclist)):
            spec=self[speclist[j]]
            rhats_spec=specrhatsarray[:,j]
            worst_rhat_spec=rhats_spec[(1-rhats_spec)**2==max((1-rhats_spec)**2)][0]
            spec.rhat=worst_rhat_spec
        return(worst_rhat)
    
    def histplot(self,ax,**kwargs):
        """
        Plots a histogram of the site level paleointensity estimate.

        Inputs
        ------
        **kwargs:
        arguments to be passed to the histogram plot

        Returns
        -------
        None
        """
        ax.hist(extract_values(self.fit,'int_site'),bins=100,color='skyblue',density=True)
        minB,maxB=np.percentile(extract_values(self.fit,'int_site'),(2.5,97.5))
        ax.plot([minB,maxB],[0,0],'k',linewidth=4)
        ax.set_xlabel('Intensity ($\mu$T)')
        ax.set_ylabel('Probability Density')
class Specimen():
    """
    Specimen from a given site or sample SpecimenCollection object.

    Parameters
    ----------

    parentCollection: SpecimenCollection object
    Site/Sample the specimen is derived from.

    specimenName: string
    Name of specimen
    """
    def __init__(self,parentCollection,specimenName):

        #Inherent properties
        self.parentCollection=parentCollection #Site or sample this specimen belongs to
        self.name=specimenName
        self.data=parentCollection.data[parentCollection.data.specimen==specimenName]
         
        self.methcodes='IE-BICEP' #Appended to when saving.
        self.extracolumnsdict={} #Extra columns for e.g. corrections
        self.rhat=1.

        #Important constants for the specimen
        self.B_lab=self.data.B_lab.iloc[0]*1e6
        self.NRM0=self.data.NRM.iloc[0]
        self.pTRMmax=max(self.data.PTRM)
        self.temps=self.data.temp_step.unique()

        #Try importing from redo file. Otherwise initiliaze to default interpretation using all measurements
        redo=parentCollection.parentData.redo
        try:
            #Interpretation temperatures
            self.lowerTemp=float(redo.loc[redo[0]==specimenName,1].iloc[0])
            self.upperTemp=float(redo.loc[redo[0]==specimenName,2].iloc[0])
            if self.lowerTemp==self.upperTemp:
                self.active=False #Used for BiCEP GUI- set to false if specimen excluded from analysis
            else:
                self.active=True

            #Saved interpretation temperatures- these are saved when the BiCEP_fit method is run.
            self.savedLowerTemp=float(redo.loc[redo[0]==specimenName,1].iloc[0])
            self.savedUpperTemp=float(redo.loc[redo[0]==specimenName,2].iloc[0])
        except:
            #Interpretation temperatures
            self.lowerTemp=min(self.temps)
            self.upperTemp=max(self.temps)

            #Saved interpretation temperatures- these are saved when the BiCEP_fit method is run.
            self.savedLowerTemp=min(self.temps)
            self.savedUpperTemp=max(self.temps)
            self.active=True

        #Definitions of Thellier Experiment Measurements (for plotting)
        self.IZZI=self.data[(self.data.steptype=='IZ')|(self.data.steptype=='ZI')]
        self.P=self.data[self.data.steptype=='P']
        self.T=self.data[self.data.steptype=='T']
        self.IZZI_trunc=self.IZZI[(self.IZZI.temp_step>=self.lowerTemp)&(self.IZZI.temp_step<=self.upperTemp)]
        self.IZ=self.IZZI_trunc[self.IZZI_trunc.steptype=='IZ']
        self.ZI=self.IZZI_trunc[self.IZZI_trunc.steptype=='ZI']
        self.saved_IZZI_trunc=self.IZZI[(self.IZZI.temp_step>=self.lowerTemp)&(self.IZZI.temp_step<=self.upperTemp)]
        self.saved_IZ=self.saved_IZZI_trunc[self.IZZI_trunc.steptype=='IZ']
        self.saved_ZI=self.saved_IZZI_trunc[self.IZZI_trunc.steptype=='ZI']

        #Definitions of Zijderveld Measurements (for plotting)
        self.NRM_dirs=self.IZZI.loc[:,'NRM_x':'NRM_z'].values
        self.NRM_trunc_dirs=self.IZZI_trunc.loc[:,'NRM_x':'NRM_z'].values

        #SPD parameters/PCA fit to direction
        self.drat=get_drat(self.IZZI,self.IZZI_trunc,self.P[(self.P.temp_step<=self.upperTemp)].iloc[:-1])
        pca=PCA(n_components=3)
        try:
            self.pca=pca.fit(self.NRM_trunc_dirs)
        except:
            self.pca=pca.fit(self.NRM_dirs)
        self.mad=get_mad(self.IZZI_trunc,self.pca)
        self.dang=get_dang(self.NRM_trunc_dirs,self.pca)

    def __repr__(self):
        return('Specimen '+self.name+'in '+self.parentCollection.key+' '+self.parentCollection.name)

    def change_temps(self,lowerTemp,upperTemp):
        """
        Changes temperature range (interpretation for specimen).
        Recalculates SPD statistic and PCA for said specimen.

        Inputs
        ------
        lowerTemp: float
        Lower temperature (inclusive) for interpretation

        upperTemp: float
        Upper temperature (inclusive) for interpretation

        Returns
        -------
        None
        """
        
        self.lowerTemp=lowerTemp
        self.upperTemp=upperTemp
        self.IZZI_trunc=self.IZZI[(self.IZZI.temp_step>=self.lowerTemp)&(self.IZZI.temp_step<=self.upperTemp)]
        self.IZ=self.IZZI_trunc[self.IZZI_trunc.steptype=='IZ']
        self.ZI=self.IZZI_trunc[self.IZZI_trunc.steptype=='ZI']
        self.drat=get_drat(self.IZZI,self.IZZI_trunc,self.P[(self.P.temp_step<=self.upperTemp)].iloc[:-1])
        pca=PCA(n_components=3)
        self.NRM_trunc_dirs=self.IZZI_trunc.loc[:,'NRM_x':'NRM_z'].values
        try:
            self.pca=pca.fit(self.NRM_trunc_dirs)
        except:
            self.pca=pca.fit(self.NRM_dirs)
        self.mad=get_mad(self.IZZI_trunc,self.pca)
        self.dang=get_dang(self.NRM_trunc_dirs,self.pca)

    def save_changes(self):
        """
        Commits temperature changes for use with the BiCEP method

        Inputs
        ------
        None

        Returns
        -------
        None
        """
        self.savedLowerTemp=self.lowerTemp
        self.savedUpperTemp=self.upperTemp
        self.saved_IZZI_trunc=self.IZZI_trunc
        self.saved_IZ=self.IZ
        self.saved_ZI=self.ZI
        if type(self.parentCollection.parentData.redo)==type(None):
            redo=pd.DataFrame({0:[],1:[],2:[]})
        else:
            redo=self.parentCollection.parentData.redo
        if self.name in redo[0].unique():
            redo.loc[redo[0]==self.name,1]=self.lowerTemp
            redo.loc[redo[0]==self.name,2]=self.upperTemp
        else:
            redo=redo.append([[self.name,self.lowerTemp,self.upperTemp]])
        self.parentCollection.parentData.redo=redo
        redo.to_csv('thellier_gui.redo',header=None,index=False,sep=' ')

    def plot_arai(self,ax,temps=True):
        """
        Plots data onto the Arai plot.

        Inputs
        ------
        ax: matplotlib axis
        axis for plot to be plotted on to

        temps: bool
        if True, plots temperatures on the Arai plot

        Returns
        -------
        None
        """
        #IZZI_trunc=self.IZZI[(self.IZZI.temp_step>=self.lowerTemp)&(self.IZZI.temp_step<=self.upperTemp)]
        lines=ax.plot(self.IZZI.PTRM/self.NRM0,self.IZZI.NRM/self.NRM0,'k',linewidth=1)
        emptydots=ax.plot(self.IZZI.PTRM/self.NRM0,self.IZZI.NRM/self.NRM0,'o',markerfacecolor='None',markeredgecolor='black',label='Not Used')
        ptrm_check=ax.plot(self.P.PTRM/self.NRM0,self.P.NRM/self.NRM0,'^',markerfacecolor='None',markeredgecolor='black',markersize=10,label='PTRM Check')
        md_check=ax.plot(self.T.PTRM/self.NRM0,self.T.NRM/self.NRM0,'s',markerfacecolor='None',markeredgecolor='black',markersize=10)
        ax.set_ylim(0,max(self.IZZI.NRM/self.NRM0)*1.1)
        ax.set_xlim(0,self.pTRMmax/self.NRM0*1.1)
        if self.active==True:
            iz_plot=ax.plot(self.IZ.PTRM/self.NRM0,self.IZ.NRM/self.NRM0,'o',markerfacecolor='b',markeredgecolor='black',label='I step')
            zi_plot=ax.plot(self.ZI.PTRM/self.NRM0,self.ZI.NRM/self.NRM0,'o',markerfacecolor='r',markeredgecolor='black',label='Z step')
        ax.set_ylabel('NRM/NRM$_0$')
        ax.set_xlabel('pTRM/NRM$_0$')
        for temp in self.temps:
            tempRow=self.IZZI[self.IZZI.temp_step==temp]
            ax.text(tempRow.PTRM/self.NRM0,tempRow.NRM/self.NRM0,str(temp-273),alpha=0.5)
    def plot_zijd(self,ax,temps=True):
        """
        Plots data onto the Zijderveld plot. Does not fit a line to this data.

        Inputs
        ------
        ax: matplotlib axis
        axis for plot to be plotted on to

        temps: bool
        if True, plots temperature values as text on plot.

        Returns
        -------
        None
        """
        #Get the NRM data for the specimen
        #Plot axis
        ax.axvline(0,color='k',linewidth=1)
        ax.axhline(0,color='k',linewidth=1)

        #Plot NRM directions
        ax.plot(self.NRM_dirs[:,0],self.NRM_dirs[:,1],'k')
        ax.plot(self.NRM_dirs[:,0],-self.NRM_dirs[:,2],'k')

        #Plot NRM directions in currently selected temperature range as closed symbols
        ax.plot(self.NRM_trunc_dirs[:,0],self.NRM_trunc_dirs[:,1],'ko')
        ax.plot(self.NRM_trunc_dirs[:,0],-self.NRM_trunc_dirs[:,2],'rs')

        #Plot open circles for all NRM directions as closed symbols
        ax.plot(self.NRM_dirs[:,0],self.NRM_dirs[:,1],'o',markerfacecolor='None',markeredgecolor='k')
        ax.plot(self.NRM_dirs[:,0],-self.NRM_dirs[:,2],'s',markerfacecolor='None',markeredgecolor='k')
        length, vector=self.pca.explained_variance_[0], self.pca.components_[0]
        vals=self.pca.transform(self.NRM_trunc_dirs)[:,0]
        v = np.outer(vals,vector)

        #Plot PCA line fit
        ax.plot(self.pca.mean_[0]+v[:,0],self.pca.mean_[1]+v[:,1],'g')
        ax.plot(self.pca.mean_[0]+v[:,0],-self.pca.mean_[2]-v[:,2],'g')
        if temps==True:
            for i in range(len(self.temps)):
                ax.text(self.NRM_dirs[i,0],-self.NRM_dirs[i,2],str(self.temps[i]-273),alpha=0.5)

        ax.set_xlabel('x, $Am^2$')
        ax.set_ylabel('y,z, $Am^2$')
        ax.axis('equal')
        ax.relim()

    def BiCEP_prep(self):
        """
        Returns the needed data for a paleointensity interpretation to
        perform the BiCEP method, calculates all corrections for a specimen.
        Performs scaling on the PTRM and NRM data. It performs the Taubin SVD circle fit
        to find the maximum likelihood circle fit to initialize the BiCEP method sampler.

        Inputs
        ------
        None

        Returns
        -------
        minPTRM: float
        Minimum scaled pTRM

        maxNRM: float
        Minimum scaled NRM

        PTRMmax: float
        Maximum total pTRM (scaled)

        k: float
        Best fitting k value using Taubin circle fit.

        phi: float
        Best fitting phi value using Taubin circle fit

        dist_to_edge: float
        Best fitting D value using Taubin circle fit.

        sigma: float
        Best fitting sigma value using Taubin circle fit.

        PTRMS: numpy.ndarray()
        Scaled and translated pTRM values

        NRMS: numpy.ndarray()
        Scaled and translated NRM values.
        """

        #Calculate Anisotropy Correction:
        if len(self.IZZI.dropna(subset=['s_tensor']))>0:
            c=calculate_anisotropy_correction(self.saved_IZZI_trunc)
            self.extracolumnsdict['int_corr_aniso']=c
            #Get method code depending on anisotropy type (AARM or ATRM)
            self.methcodes+=self.IZZI['aniso_type'].iloc[0]
        else:
            c=1

        #Get Cooling Rate Correction
        if self.IZZI.correction.iloc[0]!=1:
            self.methcodes+=':DA-CR-TRM' #method code for cooling rate correction
            self.extracolumnsdict['int_corr_cooling_rate']=self.IZZI.correction.iloc[0]


        #Calculate nonlinear TRM Correction
        if len(self.IZZI.dropna(subset=['NLT_beta']))>0:
            self.methcodes+=':DA-NL' #method code for nonlinear TRM correction
            total_correction=calculate_NLT_correction(self.saved_IZZI_trunc,c) #total correction (combination of all three corrections)
            self.extracolumnsdict['int_corr_nlt']=total_correction/(c*self.IZZI.correction.iloc[0]) #NLT correction is total correction/original correction.
        else:
            total_correction=c*self.IZZI.correction.iloc[0]

        #Converting Arai plot data to useable form
        NRMS=self.saved_IZZI_trunc.NRM.values/self.NRM0
        PTRMS=self.saved_IZZI_trunc.PTRM.values/self.NRM0/total_correction #We divide our pTRMs by the total correction, because we scale the pTRM values so that the maximum pTRM is one, this doesn't affect the fit and just gets scaled back when converting the circle tangent slopes back to intensities as would be expected, but it's easier to apply this here.

        PTRMmax=max(self.IZZI.PTRM/self.NRM0/total_correction) #We scale by our maximum pTRM to perform the circle fit.
        line=bestfit_line(self.IZZI.PTRM/self.NRM0/total_correction,self.IZZI.NRM/self.NRM0) #best fitting line to the pTRMs

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

        return(minPTRM,minNRM,PTRMmax,k,phi,dist_to_edge,sigma,PTRMS,NRMS)

    def plot_circ(self,ax,legend=False,linewidth=2,title=None,tangent=False):
        """
        Plots circle fits sampled from the posterior distribution
        (using the BiCEP method) to the Arai plot data. Plots tangent
        to the circle as a slope if tangent=True

        Inputs
        ------
        ax: matplotlib axis
        axis to be used for plot.

        legend: bool
        If True plots a legend.

        linewidth: float
        Width of circle fit lines on plot

        title: str
        Title for plot

        tangent: bool
        If set to True, plots best fitting tangent to circle.
        """

        #Get information on maximum pTRM for rescaling of circle
        fit=self.parentCollection.fit
        speclist=np.array([specimen for specimen in self.parentCollection.specimens.keys() if self.parentCollection.specimens[specimen].active==True])
        try:
            i=np.where(speclist==self.name)[0][0]
        except:
            return
        if fit!=None:
            minNRM=min(self.saved_IZZI_trunc.NRM/self.NRM0)
            minPTRM=min(self.saved_IZZI_trunc.PTRM/self.NRM0)


            #Parameters for the circle fit
            Rs=extract_values(fit,'R')[i]
            x_cs=extract_values(fit,'x_c')[i]
            y_cs=extract_values(fit,'y_c')[i]
            c=np.random.choice(range(len(Rs)),100)
            thetas=np.linspace(0,2*np.pi,1000)
            NRM0=self.NRM0

            #Circle x and y values for circle plot.
            xs=x_cs[c][:,np.newaxis]*self.pTRMmax/self.NRM0+minPTRM+Rs[c][:,np.newaxis]*np.cos(thetas)*self.pTRMmax/self.NRM0
            ys=y_cs[c][:,np.newaxis]+minNRM+Rs[c][:,np.newaxis]*np.sin(thetas)

            #Plot Circles
            ax.plot(xs.T,ys.T,'-',color='lightgreen',alpha=0.2,linewidth=linewidth,zorder=-1);
            ax.plot(100,100,'-',color='lightgreen',label='Circle Fits');

            #Find tangents to the circle:
            if tangent==True:
                phis=extract_values(fit,'phi')[i]
                dists=extract_values(fit,'dist_to_edge')[i]
                slope_ideal=-1/np.tan(np.median(phis))/self.pTRMmax*self.NRM0
                x_i=np.median(dists)*np.cos(np.median(phis))*self.pTRMmax/self.NRM0+minPTRM
                y_i=np.median(dists)*np.sin(np.median(phis))+minNRM

                ax.plot(x_i,y_i,'ko')
                c=y_i-slope_ideal*x_i
                d=-c/slope_ideal
                ax.plot([0,d],[c,0],'k',linestyle='--')

            #Add legend and title to plot
            if legend==True:
                ax.legend(fontsize=10);
            if title!=None:
                ax.set_title(title,fontsize=20,loc='left')

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
            logstring+=s+ '  has a tail check with no first zero field step - check input file! for step'+str( temp - 273.)+'\n'

#
# final check
#
    if len(first_Z) != len(first_I):
        logstring+=" Something wrong with this specimen! Better fix it or delete it "
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
    logstring=''
    for specimen in specimens:
            logstring+='Working on: '+specimen+'\n'
            clear_output(wait=True)
            print('Working on: '+specimen)
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
                                logstring+='Working on: '+specimen+'\n'
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
                                    logstring+='PTRM tail check at '+str(i)+'K has no corresponding zero field measurement, ignoring'+'\n'
                                newarray=np.array([specarray[temp_step!=diff],sample[temp_step!=diff],site[temp_step!=diff],NRM[temp_step!=diff],PTRM,NRM_x[temp_step!=diff],NRM_y[temp_step!=diff],NRM_z[temp_step!=diff],PTRM_x,PTRM_y,PTRM_z,NRM_sigma[temp_step!=diff],PTRM_sigma,B_lab[temp_step!=diff],steptype[temp_step!=diff],temp_step[temp_step!=diff]])
                                data_array=np.concatenate((data_array,newarray),axis=1)
                    else:
                        logstring+=specimen+' in site '+sitename[0]+' Not included, not a thellier experiment'+'\n'
                else:
                    logstring+=specimen+' in site '+sitename[0]+' Not included, demagnetization not completed'+'\n'
            except:
                logstring+='Something went wrong with specimen '+specimen+'. Could not convert from MagIC format'+'\n'
    temps=pd.DataFrame(data_array.T,columns=['specimen','sample','site','NRM','PTRM','NRM_x','NRM_y','NRM_z','PTRM_x','PTRM_y','PTRM_z','NRM_sigma','PTRM_sigma','B_lab','steptype','temp_step'])
    return(temps)

def generate_arai_plot_table(outputname):
    """
    Generates a DataFrame with Thellier Data for a Dataset, stores it as a csv.

    Inputs
    ------
    outputname: (str)
    name of file to output (no extension)

    Returns
    -------
    None
    """
    #This cell constructs the 'measurements' dataframe with samples and sites added
    logstring=""
    status,measurements=cb.add_sites_to_meas_table('./')
    measurements=measurements[measurements.specimen.str.contains('#')==False]
    measurements_old=measurements
    measurements=measurements[measurements.experiment.str.contains('LP-PI-TRM')]
    temps=convert_intensity_measurements(measurements)
    clear_output(wait=True)

    temps['correction']=1
    temps['s_tensor']=np.nan
    temps['aniso_type']=np.nan

    spec=pd.read_csv('specimens.txt',skiprows=1,sep='\t')

    #Create the anisotropy tensors if they don't already exist.
    logstring+="Couldn't find Anisotropy Tensors, Generating..."+'\n'

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
            meas_val['magn_moment']=meas_val['magn_moment'].astype(float)
            meas_val['treat_dc_field']=meas_val['treat_dc_field'].astype(float)
            ab,cov = curve_fit(NLTsolver, meas_val['treat_dc_field'].values*1e6,
                               meas_val['magn_moment'].values/meas_val['magn_moment'].iloc[-1],
                               p0=(max(meas_val['magn_moment']/meas_val['magn_moment'].iloc[-1]),1e-2))
            temps.loc[temps.specimen==specimen,'NLT_beta']=ab[1]
        except RuntimeError:
            logstring+="-W- WARNING: Can't fit tanh function to NLT data for "+specimen+'\n'

    #Get the cooling rate correction
    try:
        meas_cool=measurements_old[measurements_old.method_codes.str.contains('CR-TRM')].dropna(subset=['description'])
        meas_cool=meas_cool[meas_cool.method_codes.str.contains('LT-T-Z')==False] #Ignores zero field cooling rate measurements
        samples=pd.read_csv('samples.txt',skiprows=1,sep='\t')
        samples=samples.dropna(subset=['cooling_rate']) #Get only things with cooling rates from sample table
        for specimen in meas_cool.specimen.unique():
            specframe=meas_cool[meas_cool.specimen==specimen]
            vals=specframe.description.str.split(':').values #Lab cooling rates used in "description" column.
            crs=np.array([])
            for val in vals:
                crs=np.append(crs,float(val[1]))
            magn_moments=specframe['magn_moment'].astype(float).values
            avg_moment=np.mean(magn_moments[crs==max(crs)])
            norm_moments=magn_moments/avg_moment
            croven=max(crs)
            crlog=np.log(croven/crs)
            try:
                specframe['cooling_rate']=specframe.cooling_rate.astype(float) #Original cooling rate from samples table
            except AttributeError:
                try:
                    m,c=np.polyfit(crlog,norm_moments,1)
                    sample=specframe['sample'].iloc[0]
                    cr_real=samples[samples['sample']==sample].cooling_rate.values/5.256e+11
                    cr_reallog=np.log(croven/cr_real)
                    cfactor=1/(c+m*cr_reallog)[0]
                    temps.loc[temps.specimen==specimen,'correction']*=cfactor
                except AttributeError:
                    logstring+='Cooling rate correction for specimen '+specimen+' could not be calculated, original cooling rate unknown. Please add the original cooling rate (K/min) to a cooling_rate column in the specimens table. \n'
                except:
                    logstring+='Something went wrong with estimating the cooling rate correction for specimen '+specimen+ '. Check that you used the right cooling rate.'+'\n'
    except KeyError:
        logstring+="Measurements file does not contain a description for cooling rate corrections. Ignoring corrections. \n"
    #Save the dataframe to output.
    logfile=open("thellier_convert.log","w")
    logfile.write(logstring)
    logfile.close()
    clear_output(wait=True)
    print('Data conversion finished- check thellier_convert.log for errors')
    temps=temps.dropna(subset=['site'])
    temps.to_csv(outputname+'.csv',index=False)


def run_gui():
    """
    Main function for the BiCEP GUI

    Inputs:
    ------
    None

    Returns:
    -------
    None
    """
    from ipyfilechooser import FileChooser
    def display_specimen_ring():
        """
        Displays a red circle around the currently selected
        specimen in the site plot of BiCEP GUI

        Inputs:
        -------
        None

        Returns
        -------
        None
        """
        fit=thellierData[site_wid.value].fit
        if thellierData[site_wid.value].artist!=None:
            thellierData[site_wid.value].artist[0].set_marker(None)
        currspec=specimen_wid.value
        speclist=np.array([specimen for specimen in specimen_wid.options if thellierData[site_wid.value][specimen].active==True])
        specindex=np.where(speclist==currspec)
        specindex=specindex[0][0]
        ks=extract_values(fit,'k')[specindex]
        int_reals=extract_values(fit,'int_real')[specindex]
        thellierData[site_wid.value].artist=ax_2[0].plot(np.median(ks),np.median(int_reals),'o',markeredgecolor='r',markerfacecolor='None')


    def display_specimen_plots():
        """
        Displays specimen level plots on the BiCEP GUI

        Inputs:
        -------
        None

        Returns:
        --------
        None
        """
        ax[0].cla()
        ax[1].cla()
        thellierData[site_wid.value][specimen_wid.value].change_temps(lower_temp_wid.value+273,upper_temp_wid.value+273)
        thellierData[site_wid.value][specimen_wid.value].plot_circ(ax[0])
        thellierData[site_wid.value][specimen_wid.value].plot_arai(ax[0])
        thellierData[site_wid.value][specimen_wid.value].plot_zijd(ax[1])
        madbox.description='MAD: %1.2f'%thellierData[site_wid.value][specimen_wid.value].mad
        dangbox.description='DANG: %1.2f'%thellierData[site_wid.value][specimen_wid.value].dang
        dratbox.description='DRAT: %1.2f'%thellierData[site_wid.value][specimen_wid.value].drat
        rhat=thellierData[site_wid.value][specimen_wid.value].rhat
        rhatbox.description='R_hat: %1.2f'%thellierData[site_wid.value][specimen_wid.value].rhat
        if (rhat==None)|(0.9<rhat<1.1):
            rhatbox.button_style='info'
        else:
            rhatbox.button_style='danger'
        fig.tight_layout()
        ax[1].relim()

    def on_change(change):
        """
        Update GUI on changing one of our site, specimen, temperature dropdowns.

        Inputs:
        -------
        change: Dropdown change object
        Gives us information about which object was changed (owner),
        the type of change (name, either value for a value change,
        or options for all options changed),and the new value (new).
        Note that these attributes are very important to avoid repeating
        many operations, as the changing the site widget's value changes
        the specimen widgets options, which then changes it's options.
        This is the reason for the numerous if statements in this function.

        Returns:
        --------
        None
        """
        #If we're changing the site dropdown, we need to replot the site plots and change the specimen options
        if (change.owner==site_wid)&(change.name=='value'):
            specimen_wid.options=np.sort(list(thellierData[site_wid.value].specimens.keys()))
            fit=thellierData[site_wid.value].fit
            try:
                display_sampler_diags(fit)
            except:
                pass
            display_site_plot(fit)
            

        #If we're changing the specimen dropdown, we need to update the temperature steps.
        if (change.owner==specimen_wid)&(change.name=='value'):
            lower_temp_wid.options=thellierData[site_wid.value][change.new].temps-273
            upper_temp_wid.options=thellierData[site_wid.value][change.new].temps-273
            checkbox.value= not thellierData[site_wid.value][change.new].active
            #We need to change the plot to account for saved temperature steps if there are any.
            if (lower_temp_wid.value!=thellierData[site_wid.value][change.new].savedLowerTemp-273)|(upper_temp_wid.value!=thellierData[site_wid.value][change.new].savedUpperTemp-273):
                #This is fiddly, but it prevents event loop from moving on after changing value
                lower_temp_wid.unobserve(on_change)
                upper_temp_wid.unobserve(on_change)
                lower_temp_wid.value=thellierData[site_wid.value][change.new].savedLowerTemp-273
                upper_temp_wid.value=thellierData[site_wid.value][change.new].savedUpperTemp-273
                upper_temp_wid.observe(on_change)
                lower_temp_wid.observe(on_change)
                display_specimen_plots()
            #Additionally, we need to make sure the plot changes if the temperature steps were the exact same as last time.
            else:
                display_specimen_plots()
            #Finally, we need to display a ring around the specimen for the site level plot
            try:
                display_specimen_ring()
            except:
                pass

        #If we're changing the specimen plot, we display a red circle around the currently selected specimen on the site plot
        #if (change.owner==specimen_wid):
            #display_specimen_ring()

        if (change.name=='value')&((change.owner==lower_temp_wid)|(change.owner==upper_temp_wid)):
            display_specimen_plots()

    def save_temps(a):
        """
        Saves changes to specimen temperatures

        Inputs:
        ------
        a: Button pressed object
        has no practical use.

        Returns:
        -------
        None
        """
        thellierData[site_wid.value][specimen_wid.value].save_changes();

    def get_sampler_diags(site):
        """
        Returns useful sampler diagnostics for a particular MCMC fit with pystan

        Inputs
        ------
        fit: StanFit object
        model fit to site/sample

        Returns
        -------
        rhat_worst: float
        worst rhat of all parameters

        n_eff_int_site: float
        Effective number of pseudosamples of B_anc
        """
        rhat_worst=thellierData[site].get_specimen_rhats()
        n_eff_int_site=int(az.ess(thellierData[site].fit.posterior)['int_site'].values*1)
        return rhat_worst,n_eff_int_site

    def display_sampler_diags(fit):
        """
        Displays the worst R_hat and n_eff, B_anc
        and Category or Grade for the BiCEP fit

        Inputs
        ------
        fit: StanFit object
        model fit to site/sample

        Returns:
        --------
        None
        """
        rhat_worst,n_eff_int_site=get_sampler_diags(site_wid.value)
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
        int_sites=extract_values(fit,'int_site')
        cs=extract_values(fit,'c')
        minB,maxB=np.percentile(int_sites,(2.5,97.5),axis=0)
        banclabel.description='B_anc %3.1f'%minB+'- %3.1f'%maxB+' μT'
        cdiff=np.diff(np.percentile(cs,(2.5,97.5),axis=0))/np.percentile(int_sites,50)
        Bdiff=np.diff([minB,maxB])/np.percentile(int_sites,50)

        if (cdiff>=1)&(Bdiff>=0.4):
            gradelabel.description='Category: D'
            if(extract_values(fit,'k').shape[1])<5:
                gradelabel.button_style='warning'
            else:
                gradelabel.button_style='danger'
        elif(cdiff<1)&(Bdiff>=0.4):
            gradelabel.description='Category: C'
            gradelabel.button_style='warning'
        elif(cdiff>=1)&(Bdiff<0.4):
            gradelabel.description='Category: B'
            gradelabel.button_style='success'
        elif(cdiff<1)&(Bdiff<0.4):
            gradelabel.description='Category: A'
            gradelabel.button_style='success'


    def get_site_dist(a):
        """
        Runs the MCMC sampler and updates the GUI

        Inputs:
        ------
        a: Button pressed object
        has no practical use.

        Returns:
        -------
        None
        """
        process_wid.description='Processing..'

        if method_wid.value=='Slow, more accurate':
            model=model_circle_slow
        elif method_wid.value=='Fast, less accurate':
            model=model_circle_fast

        thellierData[site_wid.value].BiCEP_fit(model=model,n_samples=n_samples_wid.value)
        fit=thellierData[site_wid.value].fit
        display_sampler_diags(fit)

        #display_specimen_ring()
        display_site_plot(fit)

        process_wid.description='Process Site Data'
        ax[0].cla()
        thellierData[site_wid.value][specimen_wid.value].plot_circ(ax[0])
        thellierData[site_wid.value][specimen_wid.value].plot_arai(ax[0])
        thellierData[site_wid.value].get_specimen_rhats()

    def run(a):
        """
        Runs the GUI after selecting a file.

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        global thellierData
        run_wid.description='Converting Data...'
        thellierData=ThellierData(newfile_wid.selected_filename)
        run_wid.description='Preparing GUI...'
        site_wid.options=np.sort(list(thellierData.collections.keys()))
        specimen_wid.options=np.sort(list(thellierData[site_wid.value].specimens.keys()))

        lower_temp_wid.options=thellierData[site_wid.value][specimen_wid.value].temps-273
        upper_temp_wid.options=thellierData[site_wid.value][specimen_wid.value].temps-273
        lower_temp_wid.value=thellierData[site_wid.value][specimen_wid.value].lowerTemp-273
        upper_temp_wid.value=thellierData[site_wid.value][specimen_wid.value].upperTemp-273

        site_wid.observe(on_change)
        specimen_wid.observe(on_change)
        lower_temp_wid.observe(on_change)
        upper_temp_wid.observe(on_change)
        save_wid.on_click(save_temps)
        checkbox.observe(activate_deactivate)

        display_specimen_plots()
        process_wid.disabled=False
        save_wid.disabled=False
        savetables.disabled=False
        figsave.disabled=False
        newfile_wid.disabled=True
        checkbox.disabled=False
        run_wid.description='Running'
        run_wid.disabled=True
        savenetcdf.disabled=False




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


    def display_site_plot(fit):
        """
        Displays the site plots for BiCEP GUI

        Inputs
        ------
        fit: StanFit object
        BiCEP fit for that site/sample

        Returns
        -------
        None
        """
        ax_2[0].cla()
        ax_2[1].cla()
        try:
            thellierData[site_wid.value].regplot(ax_2[0])
            int_sites=extract_values(fit,'int_site')
            int_reals=extract_values(fit,'int_real')
            ks=extract_values(fit,'k')
            ax_2[0].axhline(np.median(int_sites),color='k')
            ax_2[1].axhline(np.median(int_sites),color='k')
            ax_2[1].hist(int_sites,color='skyblue',bins=100,density=True,orientation='horizontal')
            ax_2[0].set_ylim(min(np.percentile(int_reals,2.5,axis=0))*0.9,max(np.percentile(int_reals,97.5,axis=0))*1.1)
            ax_2[0].set_xlim(min(min(np.percentile(ks,2.5,axis=0))*1.1,min(np.percentile(ks,2.5,axis=0))*0.9),max(max(np.percentile(ks,97.5,axis=0))*1.1,max(np.percentile(ks,97.5,axis=0))*0.9))
            ax_2[1].set_ylabel('$B_{anc}$')
            ax_2[1].set_xlabel('Probability Density')
            try:
                display_specimen_ring()
            except:
                pass
        except:
            rhatlabel.description='R_hat:'
            nefflabel.description='n_eff:'
            banclabel.description='B_anc:'
            gradelabel.description='Category: '
            banclabel.button_style='info'
            nefflabel.button_style='info'
            rhatlabel.button_style='info'
            gradelabel.button_style='info'
        fig_2.tight_layout();

    def save_magic_tables(a):
        """
        Saves data from the currently displayed site to the GUI

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        try:
            thellierData[site_wid.value].save_magic_tables()
        except:
            pass

    def save_figures(a):
        """
        Saves figures from GUI to file

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """

        objdict={'Specimen Plot':fig,'Site Plot':fig_2}
        value={'Specimen Plot':specimen_wid.value,'Site Plot':site_wid.value}
        objdict[figchoice.value].savefig(value[figchoice.value]+'_BiCEP_fit.'+figformats.value)


    def enablerun(a):
        """
        Enables running the GUI after choosing a file

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        run_wid.disabled=False

    def activate_deactivate(a):
        """
        Function that excludes/includes a specimen depending on activation/deactivation

        Inputs:
        ------
        a: interact object
        Has no practical use

        Returns
        -------
        None
        """
        thellierData[site_wid.value][specimen_wid.value].active= not checkbox.value
        if thellierData[site_wid.value][specimen_wid.value].active == False:
            thellierData[site_wid.value][specimen_wid.value].change_temps(min(thellierData[site_wid.value][specimen_wid.value].temps),min(thellierData[site_wid.value][specimen_wid.value].temps))
            thellierData[site_wid.value][specimen_wid.value].save_changes()
    
    
    def save_to_netcdf(a):
        """
        Function that saves site fit to netCDF

        Inputs:
        ------
        a: interact object
        Has no practical use

        Returns
        -------
        None
        """
        try:
            thellierData[site_wid.value].fit.to_netcdf(site_wid.value+'.nc')
        except:
            pass


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
    save_wid=widgets.Button(description='Save Temperatures',disabled=True)

    newfile_wid=FileChooser('.',description='Choose File',
                              style={"description_width":"initial"},filter_pattern='*.csv')
    run_wid=widgets.Button(description='Start',
                                  style={"description_width":"initial"},disabled=True)
    newfile_wid.register_callback(enablerun)
    figsave=widgets.Button(description='Save Figures',disabled=True)
    figchoice=widgets.Dropdown(options=['Specimen Plot','Site Plot'],layout=widgets.Layout(width='20%'))
    figformats=widgets.Dropdown(description='Format:',options=['pdf','png','jpg','svg','tiff'])
    figsave.on_click(save_figures)
    run_wid.on_click(run)

    madbox=widgets.Button(description='MAD:',disabled=True)
    dangbox=widgets.Button(description='DANG:',disabled=True)
    dratbox=widgets.Button(description='DRAT:',disabled=True)
    rhatbox=widgets.Button(description='R_hat:',disabled=True)
    filebox=widgets.HBox([newfile_wid,run_wid],grid_area="filebox")
    tempbox=widgets.VBox([lower_temp_wid,upper_temp_wid],grid_area="tempbox")
    specbox=widgets.VBox([site_wid,specimen_wid],grid_area="specbox")
    checkbox=widgets.Checkbox(description='Exclude Specimen',disabled=True,indent=False,layout=widgets.Layout(width='15%'))
    savebox=widgets.HBox([save_wid,checkbox,figsave,figchoice,figformats],grid_area="savebox",layout=row_layout)
    dirbox=widgets.HBox([madbox,dangbox])
    dratrhatbox=widgets.HBox([dratbox,rhatbox])
    critbox=widgets.VBox([dirbox,dratrhatbox])
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
    rhatbox.button_style='info'
    #GUI widgets for the site processing box
    n_samples_wid=widgets.IntSlider(min=3000,max=100000,value=30000,step=1000,description='n samples')
    method_wid=widgets.Dropdown(options=['Slow, more accurate','Fast, less accurate'],description='Sampler:')
    process_wid=widgets.Button(description='Process Site Data',disabled=True)
    process_wid.on_click(get_site_dist)

    rhatlabel=widgets.Button(description='R_hat:',disabled=True)
    nefflabel=widgets.Button(description='n_eff:',disabled=True)
    banclabel=widgets.Button(description='B_anc:',disabled=True)

    gradelabel=widgets.Button(description='Category:',disabled=True)
    sampler_diag=widgets.HBox([rhatlabel,nefflabel])
    sampler_results=widgets.HBox([banclabel,gradelabel])
    sampler_buttons=widgets.VBox([sampler_diag,sampler_results])
    sampler_pars=widgets.VBox([n_samples_wid,method_wid])
    sampler_line=widgets.HBox([sampler_pars,sampler_buttons])
    banclabel.button_style='info'
    nefflabel.button_style='info'
    rhatlabel.button_style='info'
    gradelabel.button_style='info'
    siteplots=widgets.Output()
    with siteplots:
        fig_2,ax_2=plt.subplots(1,2,figsize=(6.4,3),sharey=True)
        fig_2.canvas.header_visible = False



    savetables=widgets.Button(description='Save to MagIC tables',disabled=True)
    savetables.on_click(save_magic_tables)
    savenetcdf=widgets.Button(description='Save to netCDF',disabled=True)
    sitesave=widgets.HBox([savetables,savenetcdf])
    savenetcdf.on_click(save_to_netcdf)
    

    fullbox2=widgets.VBox([process_wid,sampler_line,siteplots,sitesave],title='Site Processing')
    specpage=widgets.Accordion([fullbox])
    sitepage=widgets.Accordion([fullbox2])
    specpage.set_title(0,'Specimen Processing')
    sitepage.set_title(0,'Site Processing')
    gui=widgets.VBox([specpage,sitepage])
    display(gui)
