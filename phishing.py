import os, struct, math
#import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.mlab import bivariate_normal
import plotly
#import plotly.plotly as py
#from plotly.graph_objs import *
#import plotly.tools as tls
#from scipy.stats import multivariate_normal

import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
#from numpy import append, array, int8, uint8, zeros
import numpy as np
from numpy import uint8,array,int8
from array import array as pyarray

#from tempfile import TemporaryFile

#from pylab import *
#import scipy.sparse as sparse
import scipy.linalg as la
import scipy.io.arff as ar

import sklearn.datasets as ds
from sklearn.decomposition import PCA



def splitDF(df, frac):
    
    size = int(round(frac*len(df)))
    #print('Length: ', len(df), 'Size: ', size)
    df1 = df.sample(n=size)
    df2 = df[~df.isin(df1)].dropna()
    
    return(df1, df2)



class myBin():
    
    # class variables for bin will be initialized only by calsize
    binsize = int(0)
    p1min = 0
    p1max = 0
    p2min = 0 
    p2max = 0
    

    # Sturges Bin size determination and bin range       
    def setsizeandrange (self,p1min,p1max,p2min,p2max,samplesize=0, binsize=0):
        if (binsize !=0):
            size = binsize   
        elif (samplesize !=0):
            size = int(x=(math.log2(samplesize)+1))
        else:    
            raise Exception("Sample size or bin size needs to be specified to a bin")

        if (abs(p2max-p2min) < 0.0001) : 
            print("setrange: The ranges specified may be incorrect..."  )
        if (abs(p1max-p1min) < 0.0001) : 
            print("setrange: The ranges specified may be incorrect..."  )
            
        myBin.binsize = size
        myBin.p2min = p2min
        myBin.p2max = p2max
        myBin.p1min = p1min
        myBin.p1max = p1max
        print('\nBin Size and Range of data:')
        print('\tBinsize: %d' % size)
        print('\tP1 - Min: %.2f Max: %.2f' % (p1min,p1max))
        print('\tP2 - Min: %.2f Max: %.2f' % (p2min,p2max))

        return(size,p1min,p1max,p2min,p2max)
            
    def getsize(self):
        return(myBin.binsize)

    def getid(self, p1,p2):
        rr = np.round(0.0 + (float((myBin.binsize-1))*(p1-myBin.p1min)/(myBin.p1max-myBin.p1min)), 3)
        cc = np.round(0.0 + (float((myBin.binsize-1))*(p2-myBin.p2min)/(myBin.p2max-myBin.p2min)), 3)

        return (int(rr),int(cc))


def getFrequency(h, p1, p2):

    mb = myBin()    
    r,c = mb.getid(p1, p2) 
    count = h[r][c]

    #print('Count for height '+str(height)+' Span '+str(span)+' is:['+str(r)+','+str(c)+']= '+ str(count))  
    return(int(count),int(r),int(c))


def filterByLabel (X, T, classLabel):
    
    P = []
    count = 0
    for i in range(len(T)):
        if (T[i] == classLabel):
            P = np.append(P, X[i])
            count += 1
    P = np.array(P)
    P = np.reshape(P, (-1,2))
        
    return (P,count)



def HistClassify(p,Hn,Hp,LT,neg,pos):

    
    p = np.reshape(p, (-1,2))
    result = pd.DataFrame(index=range(0,len(p)), columns=['Ground Truth', 'Output Label', 'Probability','Class Perf'] )
    for i in range(len(LT)):    
        
        countn,rn,cn = getFrequency (Hn, p[i,0], p[i,1])
        countp,rp,cp = getFrequency (Hp, p[i,0], p[i,1])

        if ((countn+countp) == 0 or (countn == countp)):
            ClassLabel = 0
            prob = -1
            ClassPerf =  'NA'
        else:
            pclassn = float(countn)/float((countn+countp))
            pclassp = float(countp)/float((countn+countp))
            
            if (pclassn > pclassp) :
                ClassLabel = neg
                prob = pclassn
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TN'
                else:
                    ClassPerf =  'FN'
            else:
                ClassLabel = pos
                prob = pclassp
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TP'
                else:
                    ClassPerf =  'FP'
        result.loc[i]['Ground Truth'] =   LT[i]
        result.loc[i]['Output Label'] = ClassLabel
        result.loc[i]['Probability'] = prob
        result.loc[i]['Class Perf'] = ClassPerf
        
    return (result)



# classification - 1 or 3 or 0(both)
def ComputeSigmaMu(P, L, classLabel):


    X, n = filterByLabel(P,L,classLabel)
    μ=np.mean(X,axis=0,dtype=np.float64);  
    Z = X - μ
    σ = np.cov(Z,rowvar=False);
    
    '''
    print('Class Label:',classLabel)
    print('Number of samples:' ,n)
    print('Shape of P Matrix: \n', X.shape)
    print('P Matrix (X): \n', X)

    print('Shape of Mean Matrix: \n', μ.shape)
    print('Mean Matrix (μ): \n', μ)

    print('Shape of Covariance Matrix: \n', σ.shape)     
    print('Covariance Matrix (σ): \n', σ)     
    '''
    return (σ, μ, n)



def gaussianMultivariatePDF(σ, μ, X):


    
    d = len(X) # d - is the number if samples
    if d == len(μ) and (d, d) == σ.shape:
        det_sigma = np.linalg.det(σ)
        k = float(d)
        if(det_sigma != 0):
            constp  = 1.0/( np.math.pow(2.0*np.pi,k/2.0) * np.math.pow(det_sigma,1.0/2.0) )
            Z = np.matrix(X - μ)
            sigma_inv = np.linalg.inv(σ)
            pdffrac = constp*(np.math.pow(math.e, -0.5 * (Z * sigma_inv * Z.T)))
            return (pdffrac)
        else:
            raise NameError("gausianMultivariatePDF: Covariance matrix needs to be square")

    else:
        raise NameError("gausianMultivariatePDF: Invalid Matrix Shape:", len(X), len(μ), σ.shape)    


def validateGaussianMultivariatePDF():
    s = np.asarray([[9,1],[1,4]])
    m = np.asarray([1,1])
    x = np.asarray([0,0])
    
    print(np.shape(s), np.shape(m), np.shape(x))
    v = gaussianMultivariatePDF(s, m, x)
    print('From GaussianMultiVariate: ', v)
    print('Hand Calculated: ', 1.0/(2.0*math.sqrt(35.0)*np.pi*np.math.pow(math.e, (11.0/70) )))
    
    return


def plotGaussianMultivariate(P,LX,neg,pos):

 
    Sn,Mun,Nn = ComputeSigmaMu(P, LX, neg)
    Pn=np.random.multivariate_normal(Mun,Sn,size=Nn)
    X,Y = np.meshgrid(Pn[:,0], Pn[:,1])
    Z = bivariate_normal(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('viridis'))
    #cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=plt.get_cmap('viridis'))
    ax.contourf(X, Y, Z, cmap=plt.get_cmap('viridis'))
    ax.view_init(27, -21)

    Sp,Mup,Np = ComputeSigmaMu(P, LX, pos)
    Pp=np.random.multivariate_normal(Mup,Sp,size=Np)
    X,Y = np.meshgrid(Pp[:,0], Pp[:,1])
    Z = bivariate_normal(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('viridis'))
    ax.contourf(X, Y, Z, cmap=plt.get_cmap('viridis'))
    ax.view_init(27, -21)

    #ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=plt.get_cmap('viridis'))
    #cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=plt.get_cmap('viridis'))

    # Adjust the limits, ticks and view angle
    #ax.set_zlim(-0.15,0.2)
    #ax.set_zticks(np.linspace(0,0.2,5))

    plt.show()
  
    return


def BayesianClassify(P,PT,LX,LT, neg, pos):

    

    # get the Mean, covariance matrices for training data for neg and pos 
    Sn,Mun,Nn = ComputeSigmaMu(P, LX, neg)
    Sp,Mup,Np = ComputeSigmaMu(P, LX, pos)

    PT = np.reshape(PT, (-1,2))
    result = pd.DataFrame(index=range(0,len(PT)), columns=['Ground Truth', 'Output Label', 'Probability','Class Perf'] )

    # Compute the stats and save them in an dataframe
    for i in range (0, len(PT)):
        Q1 = gaussianMultivariatePDF(Sn, Mun, PT[i,0:2])
        Q2 = gaussianMultivariatePDF(Sp, Mup, PT[i,0:2])
        
        #if ((Q1+Q2) == 0 or Q1 == Q2) :
        if ((Q1+Q2) == 0) :
            ClassLabel = 0
            prob = -1
            ClassPerf =  'NA'
        else:            
            NQ = Q1/(Q1+Q2)
            PQ = Q2/(Q1+Q2)

            if (NQ>PQ) :
                ClassLabel = neg
                prob = NQ
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TN'
                else:
                    ClassPerf =  'FN'
            else:
                ClassLabel = pos
                prob = PQ
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TP'
                else:
                    ClassPerf =  'FP'
        result.loc[i]['Ground Truth'] =   LT[i]
        result.loc[i]['Output Label'] = ClassLabel
        result.loc[i]['Probability'] = prob
        result.loc[i]['Class Perf'] = ClassPerf

    return (result)




def testGaussian(P, LT, neg, pos):
 
    Sn,Mun,Nn = ComputeSigmaMu(P, LT, neg)
    Sp,Mup,Np = ComputeSigmaMu(P, LT, pos)
    
    
    Xn = np.random.multivariate_normal(Mun, Sn, 3000)
    Xp = np.random.multivariate_normal(Mup, Sp, 3000)
    
    X = np.append(Xn,Xp,axis=0)
    
    result = pd.DataFrame(index=range(0,len(X)), columns=['Ground Truth', 'Output Label', 'Probability','Class Perf'] )

    # Compute the stats and save them in an dataframe
    for i in range (0, len(X)):
        Q1 = gaussianMultivariatePDF(Sn, Mun, X[i,0:2])
        Q2 = gaussianMultivariatePDF(Sp, Mup, X[i,0:2])
        
        if ((Q1+Q2) == 0 or Q1 == Q2) :
            ClassLabel = 0
            prob = -1
            ClassPerf =  'NA'
        else:            
            NQ = Q1/(Q1+Q2)
            PQ = Q2/(Q1+Q2)

            if (NQ>PQ) :
                ClassLabel = neg
                prob = NQ
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TN'
                else:
                    ClassPerf =  'FN'
            else:
                ClassLabel = pos
                prob = PQ
                if (LT[i] == ClassLabel):
                    ClassPerf =  'TP'
                else:
                    ClassPerf =  'FP'
        result.loc[i]['Ground Truth'] =   LT[i]
        result.loc[i]['Output Label'] = ClassLabel
        result.loc[i]['Probability'] = prob
        result.loc[i]['Class Perf'] = ClassPerf
        
        

    plt.scatter(Xn[:,0],Xn[:,1], c = 'red', s=1)
    plt.scatter(Xp[:,0],Xp[:,1], c = 'blue', s=1)
    plt.title('Scatter plot of generated gaussian points blue:(+), red:(-)')
   
    plt.show()
        
        
        
        
    computeAccuracy(result)
    return (result) 
  


def plot3DCombinedHist(P, LX, pos, neg):

    mb = myBin()
    size = mb.getsize()
    H = [[0]*(size) for _ in range(size)]
    colors = [['w']*(size) for _ in range(size)]    
    H = np.array(H)
    colors = np.array(colors)
    P1 = np.array([])
    P2 = np.array([])
    for i in range (len(P)):
        r,c = mb.getid(P[i,0],P[i,1])
        P1 = np.append(P1, P[i,0])
        P2 = np.append(P2, P[i,1])   
        H[r][c] += 1 
        if LX[i] == pos:
            if colors[r][c] == 'r' :
                colors[r][c] = 'g'
            elif colors[r][c] == 'g':
                colors[r][c] = 'g'
            else:
                colors[r][c] = 'b'
        else:
            if colors[r][c] == 'b' :
                colors[r][c] = 'g'
            elif colors[r][c] == 'g':
                colors[r][c] = 'g'
            else:
                colors[r][c] = 'r'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(P1, P2, bins=size, range=[[0, size], [0, size]])

    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = H.flatten()
    colors = colors.flatten()
    ax.view_init(elev=10., azim=-75)

    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
    plt.title('3D Histogram : Positive - Blue | Negative - Red | Overlap - Green ')

    plt.show()
    return
    

def computeAccuracy(df):
    
    tpc = 0
    fpc = 0
    tnc = 0
    fnc = 0
    
    tpc = len(df[df['Class Perf'] == 'TP'])
    tnc = len(df[df['Class Perf'] == 'TN'])
    fpc = len(df[df['Class Perf'] == 'FP'])
    fnc = len(df[df['Class Perf'] == 'FN'])
    
    print('\tNumber of Samples: ', df['Output Label'].count())
    print('\tTP Count: ', tpc,'\n\tTN Count: ', tnc,'\n\tFP Count: ', fpc,'\n\tFN Count: ', fnc)
    
    
    accuracy = (tpc+tnc)/(tpc+tnc+fpc+fnc)

    print('\tAccuracy: ', accuracy)    
    return(accuracy)

def plot3Dhist(x,y,z,ibins):

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=ibins, range=[[0, ibins], [0, ibins]])

    print('hist:\n', hist)
    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = z.flatten()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    plt.show()
    return

"""
Iterate through the label and count frequncy for the labels for min or max
"""
def create2DHistogram (P, LX, classification ):

    mb = myBin()
    size = mb.getsize()
    H = [[0]*(size) for _ in range(size)]
    P1 = np.array([])
    P2 = np.array([])
    for i in range (len(P)):
        if LX[i] == classification:
            r,c = mb.getid(P[i,0],P[i,1])
            P1 = np.append(P1, P[i,0])
            P2 = np.append(P2, P[i,1])
            H[r][c] += 1 
    H = np.array(H)
    return(P1,P2,H)


def saveimgtopgm(filename, image, size=28, maxentry=255): 
   
    f = open(filename,"w+")
    # Header of PGM file
    f.write("P2  #magic\r\n%d  #image width\r\n%d  #image height\r\n%d #max entry (white). Image data follows ....\r\n" % (size,maxentry))
    # Image data
    for row in range(size):
        for col in range(size):
            tmp_d = image[row,col]
            f.write("%d\t" % tmp_d)
        f.write("\r\n")
            
    f.close()
    return


# Usage vectorimg (random.random(784)) or vectorimg(random.random(784),arrange(784))
def vectortoimg(*args,width=28,size=1):
    n= len(args)
    fig = plt.figure()
    for i,arg in enumerate(args):
        plt.subplot(1,n,i+1)
        plt.imshow(arg.reshape(width,width), interpolation='None',cmap=plt.get_cmap('gray'))
        plt.axis('off')
    fig.tight_layout(pad=0)
    fig.set_size_inches(w=n*size,h=size)
    plt.show()
    return
        
def generateScatterPlotofP(X,T,pos,neg):

    #Create 2D scatter plots with different colors for P1/P2
    print('Generate scatter plot for P1 & P2...')    
    Xn,Nn = filterByLabel (X, T, neg)
    Xp,Np = filterByLabel (X, T, pos)
    
    plt.scatter(Xn[:,0],Xn[:,1], c = 'r', s=1)
    plt.scatter(Xp[:,0],Xp[:,1], c = 'g', s=1)
    μn=np.mean(Xn,axis=0,dtype=np.float64);  
    μp=np.mean(Xp,axis=0,dtype=np.float64);  
    plt.scatter(μp[0],μp[1], color='b', s=30)
    plt.scatter(μn[0],μn[1], color='b', s=30)
    plt.title('Scatter plot of P1,P2 Red:(), Green:(-)')
    plt.annotate('μp', xy=μp, xytext=μp*1.1)
    plt.annotate('μn', xy=μn, xytext=μn*1.1)
    plt.show()
    return



    
def computeVerifyμZC (X):

    # Compute μ (row vector) and Z  
    print("\nComputing and plotting μ" )
    μ=np.mean(X,axis=0,dtype=np.float64); # axis =0 => Means of columns     
    print('\tShape of μ:', μ.shape)
    print('\tMin/Max of μ: %.2f, %.2f' % (np.amin(μ), np.amax(μ)))
    
    #print('\nVector Image of μ:\n' )

    #vectortoimg(μ)
    
    print('\nGenerate plot of μ:\n' )
    
    plt.plot(μ)
    plt.show()
    #print('Printing μ:\n', μ)
    print("\nComputing Z" )
    Z = X- μ
    print('\tShape of Z:', Z.shape)
    #print('Printing Z:\n', Z)

    # Compute Covariance matrix 
    # Note that the option rowvar=False has to be passed to the cov() function to prevent 
    # the rows from being interpreted as the variables (features). 
    # Alternatively, we could have computed C=np.cov(Z.T)

    print('Computing Covariance Matrix....')    
    C=np.cov(Z,rowvar=False);
    print('\tShape of C: ', C.shape)
    #print('Printing C:\n', C)
    # Check for Symmetry - Verify symmetry Cij = Cji

    print('\tChecking for Symmetry of Covariance Matrix....')    
    if np.array_equal(C, C.T) != True : 
        print("\tCovariance Matrix is not Symmetrical!!")
    else:
        print("\tCovariance Matrix C is Symmetrical....")

    print('\tChecking for Diagonal Elements ....')        
    cerr = 0 
    for i in range(len(C)):
            if C[i][i] < 0:
                print('Value of Diagonal Element' + str(i) + 'less than Zero')
                cerr = 1
    if cerr == 1:
        print('\tSome diagonal elements are negative' )
    else:
        print('\tDiagonal check successful')
    
    # Diplay the Covariance matrix as an Image.  If the vaules are 
    # scaled to 0-255, they can be saved in PGM format and viewed with GIMP
    print("\nPlotting Covariance matrix....");
    vectortoimg(C, width=30, size=6)          
    np.savetxt("mu.csv", μ, delimiter=",")

    return(μ,Z,C)

def computeVerifyEigen(C):


    print('\nComputing Eigenvector')

    [λnumpy,Vnumpy]=la.eigh(C);
    np.savetxt('eigen.csv', Vnumpy, delimiter = ",")
    
    print('\tShape of Numpy Eigenvector Vnumpy: ', Vnumpy.shape)
    #print('Printing Numpy Eigenvector Vnumpy:\n', Vnumpy)

    print('\tShape of Eigen Value λnumpy: ', λnumpy.shape)
    #print('Printing Eigen Value λnumpy:\n', λnumpy)

    # For notational agreement with the Lecture Notes, we will set V to its own transpose. 
    # We also need to reverse the order of eigenvectors and eigenvalues so that they are ordered 
    # in decreasing order of importance.
    λ=np.flipud(λnumpy);
    V=np.flipud(Vnumpy.T); 

    #print('\nShape of Transposed and Reordered Eigenvector V: ', V.shape)
    #print('Printing Transposed and Reordered Eigenvector V:\n', V)

    #print('\nShape of Reordered Eigen Value λ: ', λ.shape)
    #print('Printing Reordered Eigen Value λ:\n', λ)

    print ('\tV[0] Normality Check:', np.linalg.norm(V[0]))
    print ('\tV[1] Normality Check:', np.linalg.norm(V[1]))
    print ('\tV[0].V[1] Orthogonality Check: ', np.dot(V[0,:], V[1,:]))
    #print('\nPlot the Eigenvectors for v1(V[0]), v2(V[1])') 
    #vectortoimg(V[0],V[1], width=28, size=1)
    
    return(V,λ,Vnumpy,λnumpy)


def computelambdapercentages(λ):
    
    #print(λ, np.sum(λ))
    
    λp = np.zeros(np.shape(λ))
    
    for i in range(len(λ)):
        for j in range(i+1):
            λp[i] += λ[j]
            #print(i, j, λp[i], λ[j])
        λp[i] = 100*λp[i]/np.sum(λ)

    #print(λp)
    return(λp)




def visualizeExplainedVariance(eigenvals):

    tot = sum(eigenvals)
    var_exp = [(i / tot)*100 for i in sorted(eigenvals, reverse=True)]
    #print(var_exp)
    cum_var_exp = np.cumsum(var_exp)
    objects = ['%s' %i for i in range(1,len(eigenvals)+1)]
    #print(objects)
    y_pos = np.arange(len(eigenvals))
    fig, ax = plt.subplots()
    plt.bar (y_pos, var_exp, align='center', alpha=0.5)

    #ax.plot(y_pos, cum_var_exp, '--' )
    ax.plot(y_pos, cum_var_exp)
    plt.plot((0, 12), (80, 80))
    plt.plot((12, 12), (0, 80))
    
    ax.set_xticklabels(ax.get_xlabel(), rotation=0, fontsize=6)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percent')
    plt.title('Explained variance (%) of Principal components ') 
    fig.tight_layout()
    plt.show()
    return    

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=0)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

def createConfMatrix(expected, predicted, lbls=None):

    print('\nGenerating & Plotting Confusion Matrix...')

    if lbls is None:
        y_actu = pd.Series(expected, name='True Class')
        y_pred = pd.Series(predicted, name='Classified as')
    else:    
        y_actu = pd.Series(expected, name='True Class').astype("category", categories=lbls)
        y_pred = pd.Series(predicted, name='Classified as').astype("category", categories=lbls)

    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['True Class'], colnames=['Classified as'], margins=True)
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['True Class'], colnames=['Classified as'])
    print(df_confusion)


    plot_confusion_matrix(df_confusion, title='Confusion matrix')
    
    
    m = df_confusion.as_matrix()

    return(m)

def augmentFeatureVector(df):

    df.insert(0, 'w0',1)

    return(df)


def buildLinearClassifier(X, Tin):

    print('\tBuilding Linear Classifier...');

    Xa = np.asarray(augmentFeatureVector(X))
    T  = np.asarray(Tin)

    W = np.dot(np.linalg.pinv(Xa),T)       # Classifier
    
    return(W,Xa)    

def applyLinearClassifier(W, Xa):
    
    debug = 0
    print('\tApplying Classifier...');
    if (debug == 1):    
        print('Shape of W: ', W.shape)
        print('W: \n', W)
        print('Shape of Xa: ', Xa.shape)
        print('Xa: \n', Xa)

    T = np.dot(Xa,W)
    
    return(T)

def calPPV(confusionmatrix):
    print('Calculating Positive predictive value...')
    # for each column, divide the row with index of the column by sum of all the items in the column
    ppv = np.sum(confusionmatrix,axis=0)
    ppv_r = np.zeros_like(ppv, dtype=float)
    for i in range (len(confusionmatrix)):
        if (ppv[i] == 0):
            ppv_r[i] = 'NaN'
        else:
            ppv_r[i] = confusionmatrix[i][i] /ppv[i]
    print('\tPPV:', ppv_r)
    return(ppv_r)

def calculateMetric(confusionmatrix):
    
    if (np.shape(confusionmatrix) != (2,2)):
        print('Metrics are meant for Binary classifications only...')
        return (0,0,0,0)
    else:
        TP = confusionmatrix[1][1]
        FP = confusionmatrix[0][1]
        TN = confusionmatrix[0][0]
        FN = confusionmatrix[1][0]
        #print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
        accuracy = (TP+TN) /(TP+TN+FP+FN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        ppv = TP/(FP+TP)
        
        print('\nClassifier Performance Metrics' )
        print('\tAccuracy: ', accuracy, '\n\tSensitivity: ', sensitivity, '\n\tSpecificity: ', specificity, '\n\tPPV: ', ppv)
    return (accuracy,sensitivity,specificity,ppv)

def main():
    
    from sklearn.naive_bayes import GaussianNB


    print('Version 4-4-2017-V003')    
    neg = -1
    pos = 1
    
    print('Loading data from the data set....' )
    regeneratecsv = 0
    if (regeneratecsv == 1):
        data,meta = ar.loadarff(open('phishing1.arff', 'r', encoding='utf-8'))

        df = pd.DataFrame(data, dtype = int)

        dftr,dfts = splitDF(df, .70)
        dftr.reset_index(drop=True, inplace=True)
        dfts.reset_index(drop=True, inplace=True)
        dftr.to_csv('phishing_training.csv')
        dfts.to_csv('phishing_testing.csv')
    else:
        dftr= pd.DataFrame.from_csv('phishing_training.csv')
        dfts= pd.DataFrame.from_csv('phishing_testing.csv' )

    print(dftr['Result'].count(), dfts['Result'].count())
    print('Number of Positive Samples in Training Dataset: ', len(dftr[dftr['Result'] == -1]))
    print('Number of Negative Samples in Training Dataset: ', len(dftr[dftr['Result'] != -1]))
    print('Number of Positive Samples in Test Dataset: ', len(dfts[dfts['Result'] == -1]))
    print('Number of Negative Samples in Test Dataset: ', len(dfts[dfts['Result'] != -1]))
    #print(dftr.head())

    Xdf = dftr.drop('Result', axis=1)
    #Xdf = Xdf.reset_index(drop=True)
    X = np.asarray(Xdf)
    print('\tShape of X: ', np.shape(X))
    Ldf = pd.DataFrame(dtype=int)
    Ldf['Result'] = dftr['Result']
    #Ldf = Ldf.reset_index(drop=True)
    T = np.asarray(Ldf)
    print('\tShape of T: ', np.shape(T))
    #print (T)


    
    
    # Compute mean, Z and Covariance matrices and perform checks
    μ,Z,C = computeVerifyμZC (X)
    
    # Compute the Eigen values(λ) and Normalized Eigne Vectors (V)- 
    # Note: Numpy la.eigh returns the vectors as Columns in ascending order
    # Note V is an orthogonal matrix ==> Transpose=Inverse

    V,λ,Vnumpy,λnumpy = computeVerifyEigen(C)
    
    
    # Compute principlal components.  Vp - Principle Eigen Vectors (v1,v2)
    Vpb = V[0:2,:]
    
    print('Computing P...')
    P=np.dot(Z,Vpb.T);
    
    print('\tShape of P: ', P.shape )
    #print('Printing P:\n', P)
    print('\tCheck for value P-mean to be nearly zero:', P[:,0:2].mean())


    σp, μp, sizep = ComputeSigmaMu(P, T, pos)
    σn, μn, sizen = ComputeSigmaMu(P, T, neg)
    
    print('\tNp (number of positive(%d) samples): %d'% (pos,sizep))
    print('\tNn (number of negative(%d) samples): %d'% (neg,sizen))
    print('\tμp (2D mean vector derived from positive samples of P): ', μp)
    print('\tμn (2D mean vector derived from negative samples of P): ', μn)
    print('\tσp (2x2 cov matrix derived from positive samples of P):\n',σp)
    print('\tσn (2x2 cov matrix derived from negative samples of P):\n',σn)
    p1min = P[:,0].min(); p1max = P[:,0].max()
    p2min = P[:,1].min(); p2max = P[:,1].max()
    print('\tMin & Max of 1st principal components: ', p1min, p1max)
    print('\tMin & Max of 2nd principal components: ', p2min, p2max)

    # Scatter plot of P
    generateScatterPlotofP(P,T,pos,neg)

    #-------------------
    print("Analyze the explaied variance of the principal components\n")
    visualizeExplainedVariance(λ)

    # based on the 80% explained variance of Pricipal component, we 
    # pick 13 vectors
    Vp = V[0:13,:]
    P=np.dot(Z,Vp.T);
    print('\tShape of P: ', P.shape )

    # Native Bayes multifeature gaussian analysis
    print("Initiate Native Bayes multi feature analysis..")
    gnb = GaussianNB()
    gnb.fit(P,T.reshape(-1))
    
    print('Process the test data set...' )
    Xsdf = dfts.drop('Result', axis=1)
    Xsdf = Xsdf.reset_index(drop=True)
    Xs = np.asarray(Xsdf)
    print('\tShape of Xs: ', np.shape(Xs))
    Lsdf = pd.DataFrame(dtype =int)
    Lsdf['Result'] = dfts['Result']
    Lsdf = Lsdf.reset_index(drop=True)
    Ts = np.asarray(Lsdf)

    Zs = Xs-μ
    Ps = np.dot(Zs,Vp.T)
    
    print('Predicting class labels for test data...')
    Tpred = gnb.predict(Ps)

    
    len1 = len(Ts) 
    Tpred = Tpred.reshape(len1,1)
    Ts = Ts.reshape(len1,1)
    
    print("\tPCA/Naive Bayes: Number of mislabeled points out of a total %d points : %d"
      % (Ts.shape[0],(Ts != Tpred).sum()))
    
    Ts = Ts.reshape(len1)
    Tpred = Tpred.reshape(len1)
    cmm = createConfMatrix(Ts, Tpred)
    calPPV(cmm)
    calculateMetric(cmm)

    # Linear MSE minimizer
    print('\nInitiate Linear MSE minimizer...')

    Xdf = dftr.drop('Result', axis=1)
    Xdf = Xdf.reset_index(drop=True)
    Ldf = pd.DataFrame()
    Ldf['Result'] = dftr['Result']
    W, Xa = buildLinearClassifier(Xdf, Ldf)
    
    Xsdf = dfts.drop('Result', axis=1)
    Xsdf = Xsdf.reset_index(drop=True)
    Xsdf1 = augmentFeatureVector(Xsdf)
    Xsa = np.asarray(Xsdf1)
    
    Lsdf = pd.DataFrame()
    Lsdf['Result'] = dfts['Result']
    Lsdf = Lsdf.reset_index(drop=True)
    Tsi = np.asarray(Lsdf).astype(int)
    Tso = np.sign(applyLinearClassifier(W, Xsa)).astype(int)
        
    print("\tLinear MSE: Number of mislabeled points out of a total %d points : %d"
      % (Tsi.shape[0],(Tsi != Tso).sum()))

    cmm = createConfMatrix(Tsi.reshape(-1), Tso.reshape(-1))
    calPPV(cmm)
    calculateMetric(cmm)
    
    # Decision Tree
    print('\nInitiating a Decision Tree....')
    from sklearn import tree
    X = np.asarray(Xdf)
    T = np.asarray(Ldf)

    Xt = np.asarray(Xsdf)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, T)
    Tt = clf.predict(Xt).reshape(-1)
    
    Tt = Tt.reshape(len1,1)
    
    print("\tDecisison Tree: Number of mislabeled points out of a total %d points : %d"
      % (Tsi.shape[0],(Tsi != Tt).sum()))
    
    cmm = createConfMatrix(Tsi.reshape(-1), Tt.reshape(-1))
    calPPV(cmm)
    calculateMetric(cmm)    

    print('\nInitiating SVM.....')    
    from sklearn import svm
    X = np.asarray(Xdf)
    T = np.asarray(Ldf).reshape(-1)

    Xt = np.asarray(Xsdf)
    clf = svm.SVC()
    clf.fit(X, T)  

    Tt = clf.predict(Xt).reshape(len1,1)
    
    print("\tSVM: Number of mislabeled points out of a total %d points : %d"
      % (Tsi.shape[0],(Tsi != Tt).sum()))
    
    cmm = createConfMatrix(Tsi.reshape(-1), Tt.reshape(-1))
    calPPV(cmm)
    calculateMetric(cmm)

    print('\nInitiating Neural network.....')    
    '''
    Input layer number of neurons = 30 + 1 for Bias
    Output Layer number of neurons = 1
    Hidden layers = 5
    Neourons per hidden layer = 15
    '''
    from sklearn.neural_network import MLPClassifier
    X = np.asarray(Xdf)
    T = np.asarray(Ldf).reshape(-1)

    Xt = np.asarray(Xsdf)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,15), random_state=1)
    clf.fit(X, T)  

    Tt = clf.predict(Xt).reshape(len1,1)
    
    print("\tNeural Network: Number of mislabeled points out of a total %d points : %d"
      % (Tsi.shape[0],(Tsi != Tt).sum()))
    
    cmm = createConfMatrix(Tsi.reshape(-1), Tt.reshape(-1))
    calPPV(cmm)
    calculateMetric(cmm)




    return    
    

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

