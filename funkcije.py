# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:16:43 2015

@author: Ziga Spiclin

Vaja 10: Sledenje in analiza gibanja
"""

import scipy.ndimage as ni
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp
from scipy.interpolate import interpn
import matplotlib.animation as animation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import PIL.Image as im
import math 


#------------------------------------------------------------------------------
# POMOZNE FUNKCIJE
def showImage( iImage, iTitle='', iTranspose=False, iCmap=cm.Greys_r ):
    """Prikazi sliko v lastnem naslovljenem prikaznem oknu"""
    # preslikaj koordinate barvne slike    
    if len(iImage.shape)==3 and iTranspose:
        iImage = np.transpose( iImage, [1,2,0])
    plt.figure()
    if iImage.dtype.kind in ('u','i'):
        vmin_ui = np.iinfo(iImage.dtype).min
        vmax_ui = np.iinfo(iImage.dtype).max
        plt.imshow(iImage, cmap = iCmap, vmin=vmin_ui, vmax=vmax_ui)
    else:
        plt.imshow(iImage, cmap = iCmap)
    plt.axes().set_aspect('equal', 'datalim')
    plt.suptitle( iTitle )
    plt.xlabel('Koordinata x')
    plt.ylabel('Koordinata y')
    # podaj koordinate in indeks slike
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (iImage[y, x], x, y)
        except IndexError:
            return "IndexError"    
    plt.gca().format_coord = format_coord
    #plt.axes('equal') # should, but doesnt work
    plt.show()
    
def colorToGray( iImage ):
    """Pretvori v sivinsko sliko"""
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    colIdx = [iImage.shape[i] == 3 for i in range(len(iImage.shape))]
    
    if colIdx.index( True ) == 0:
        iImageG = 0.299 * iImage[0,:,:] + 0.587 * iImage[1,:,:] + 0.114 * iImage[2,:,:]
    elif colIdx.index( True ) == 1:
        iImageG = 0.299 * iImage[:,0,:] + 0.587 * iImage[:,1,:] + 0.114 * iImage[:,2,:]
    elif colIdx.index( True ) == 2:
        iImageG = 0.299 * iImage[:,:,0] + 0.587 * iImage[:,:,1] + 0.114 * iImage[:,:,2]
    
    return np.array( iImageG, dtype = iImageType )
       
def discreteConvolution2D( iImage, iKernel ):
    """Diskretna 2D konvolucija slike s poljubnim jedrom"""    
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray( iImage )
    iKernel = np.asarray( iKernel )
    #------------------------------- za hitrost delovanja
    oImage = ni.convolve( iImage, iKernel, mode='nearest' )    
    return oImage

def imageGradient( iImage ):
    """Gradient slike s Sobelovim operatorjem"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((-1,0,1),(-2,0,2),(-1,0,1)) )    
    oGx = ni.convolve( iImage, iSobel, mode='nearest' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='nearest' )
    return oGx, oGy

def decimateImage2D( iImage, iLevel ):
    """Funkcija za piramidno decimacijo"""  
#    print('Decimacija pri iLevel = ', iLevel)
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array( ((1/16,1/8,1/16),(1/8,1/4,1/8),(1/16,1/8,1/16)) )
    # glajenje slike pred decimacijo
    iImage = discreteConvolution2D( iImage, iKernel )
    # decimacija s faktorjem 2
    iImage = iImage[::2,::2]
    # vrni sliko oz. nadaljuj po piramidi
    if iLevel <= 1:
        return np.array( iImage, dtype=iImageType )
    else:
        return decimateImage2D( iImage, iLevel-1 )       
       
def interpolate1Image2D( iImage, iCoorX, iCoorY ):
    """Funkcija za interpolacijo prvega reda"""
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )    
    iCoorX = np.asarray( iCoorX )
    iCoorY = np.asarray( iCoorY )   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')
    #------------------------------- za hitrost delovanja    
    return interpn( (np.arange(dy),np.arange(dx)), iImage, \
                      np.dstack((iCoorY,iCoorX)),\
                      method='linear', bounds_error=False)\
                      .astype( iImage.dtype )    
               
def transAffine2D( iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0) ):
    """Funkcija za poljubno 2D afino preslikavo"""    
    iRot = iRot * np.pi / 180
    oMatScale = np.matrix( ((iScale[0],0,0),(0,iScale[1],0),(0,0,1)) )
    oMatTrans = np.matrix( ((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)) )
    oMatRot = np.matrix( ((np.cos(iRot),-np.sin(iRot),0),\
                          (np.sin(iRot),np.cos(iRot),0),(0,0,1)) )
    oMatShear = np.matrix( ((1,iShear[0],0),(iShear[1],1,0),(0,0,1)) )
    # ustvari izhodno matriko
    oMat2D = oMatTrans * oMatShear * oMatRot * oMatScale
    return oMat2D               
               
def transformImage( iImage, oMat2D ):
    """Preslikaj 2D sliko z linearno preslikavo"""
    # ustvari diskretno mrezo tock
    gx, gy = np.meshgrid( range(iImage.shape[1]), \
                          range(iImage.shape[0]), \
                          indexing = 'xy' )    
    # ustvari Nx3 matriko vzorcnih tock                          
    pts = np.vstack( (gx.flatten(), gy.flatten(), np.ones( (gx.size,))) ).transpose()
    # preslikaj vzorcne tocke
    pts = np.dot( pts, oMat2D.transpose() )
    # ustvari novo sliko z interpolacijo sivinskih vrednosti
    oImage = interpolate1Image2D( iImage, \
                                  pts[:,0].reshape( gx.shape ), \
                                  pts[:,1].reshape( gx.shape ) )
    oImage[np.isnan( oImage )] = 0
    return oImage 

def showVideo( oVideo, oPathXY=np.array([]) ):
    """Prikazi video animacijo poti"""
    global oVideo_t, iFrame, oPathXY_t
    fig = plt.figure()
    # prikazi prvi okvir
    iFrame = 0
    oPathXY_t = oPathXY
    oVideo_t = oVideo
    print(oVideo.shape)
    im = plt.imshow(oVideo[...,iFrame], cmap=plt.get_cmap('Greys_r'))
    # definiraj funkcijo za osvezevanje prikaza
    def updatefig(*args):
        global oVideo_t, iFrame, oPathXY_t
        iFrame = ( iFrame + 1 ) % oVideo_t.shape[-1]
        im.set_array( oVideo_t[...,iFrame] ) 
        if iFrame < oPathXY.shape[0]:
            plt.plot( oPathXY[iFrame,0], oPathXY[iFrame,1], 'xr' ,markersize=3 )    
        return im,
    # prikazi animacijo poti
    ani = animation.FuncAnimation(fig, updatefig, interval=25, blit=True)
    plt.show()  

def drawPathToFrame( oVideo, oPathXY, iFrame=1, iFrameSize=(40,40) ):    
    """Prikazi pot do izbranega okvirja"""
    oPathXY_t = oPathXY[:iFrame,:]
    showImage( oVideo[...,iFrame], 'Pot do okvirja %d' % iFrame )
    for i in range(1,oPathXY_t.shape[0]):
        plt.plot(oPathXY_t[i-1:i+1,0],oPathXY_t[i-1:i+1,1],'--r')
        if i==1 or (i%5)==0:
            plt.plot( oPathXY_t[i,0],oPathXY_t[i,1],'xr',markersize=3)
        
    dx = iFrameSize[0]/2; dy = iFrameSize[1]/2
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]+dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot( (oPathXY_t[-1,0]+dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]-dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]-dy),'-g')
   
#%% Naloga 1: Nalozi video
# za nalaganje videa boste potrebovali knjiznico ffmpeg (datoteko ffmpeg.exe),
# ki jo lahko nalozite s spletne strani https://www.ffmpeg.org/download.html
def loadVideo( iFileName, iFrameSize = (576, 720) ):
    """Nalozi video z ffmpeg orodjem"""
    import sys
    import subprocess as sp
    # ustvari klic ffmpeg in preusmeri izhod v cevovod
    command = [ 'ffmpeg',
                '-i', iFileName,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    # definiraj novo spremeljivko
    oVideo = np.array([])
    iFrameSize = np.asarray( iFrameSize )
    frameCount = 0
    # zacni neskoncno zanko
    while True:
        frameCount += 1
#        print( 'Berem okvir %d ...' % frameCount )
        print("\rBerem okvir %d ..." % frameCount, end="")
        # preberi Y*X*3 bajtov (= 1 okvir)
        raw_frame = pipe.stdout.read(np.prod(iFrameSize)*3)
        # pretvori prebrane podatke v numpy polje
        frame =  np.fromstring(raw_frame, dtype='uint8')       
        # preveri ce je velikost ustrezna, sicer prekini zanko
        if frame.size != (np.prod(iFrameSize)*3):
            print(" koncano!\n")
            break;
        # preoblikuj dimenzije in pretvori v sivinsko sliko
        frame = colorToGray( frame.reshape((iFrameSize[0],iFrameSize[1],3)) )
        # sprazni medpomnilnik        
        pipe.stdout.flush()    
        # vnesi okvir v izhodno sprememnljivko
        if oVideo.size == 0:
            oVideo = frame
            oVideo = oVideo[...,None]
        else:
            oVideo = np.concatenate((oVideo,frame[...,None]), axis=2)
    # zapri cevovod
    pipe.terminate()
    # vrni izhodno spremenljivko
    return oVideo
    
# test funkcije
if __name__ == '__main__':
    # nalozi video
    oVideo = loadVideo( 'video1.avi' )
    # prikazi prvi okvir
    plt.close('all')
    showImage( oVideo[..., 0] )
    # prikazi video
    plt.close('all')    
    showVideo( oVideo )

#%% Naloga 2: Funkcija za poravnavo z Lucas-Kanade postopkom
def regLucasKanade( iImgFix, iImgMov, iMaxIter, oPar = (0,0), iVerbose=True ):
    """Postopek poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgType = np.asarray( iImgMov ).dtype
    iImgFix = np.array( iImgFix, dtype='float' )
    iImgMov = np.array( iImgMov, dtype='float' )
    # doloci zacetne parametre preslikae
    oPar = np.array( oPar )     
    # izracunaj prva odvoda slike
    Gx, Gy = imageGradient( iImgMov )      
    # v zanki iterativno posodabljaj parametre
    for i in range( iMaxIter ):
        # doloci preslikavo pri trenutnih parametrih        
        oMat2D = transAffine2D( iTrans=oPar )        
        # preslikaj premicno sliko in sliki odvodov        
        iImgMov_t = transformImage( iImgMov, oMat2D )
        Gx_t = transformImage( Gx, oMat2D )
        Gy_t = transformImage( Gy, oMat2D )        
        # izracunaj sliko razlike in sistemsko matriko
        I_t = iImgMov_t - iImgFix
        B = np.vstack( (Gx_t.flatten(), Gy_t.flatten()) ).transpose()
        # resi sistem enacb
        invBtB = np.linalg.inv( np.dot( B.transpose(), B ) )
        dp = np.dot( np.dot( invBtB, B.transpose() ), I_t.flatten() )        
        # posodobi parametre        
        oPar = oPar + dp.flatten()           
        if iVerbose: print('iter: %d' % i, ', oPar: ', oPar)
    # doloci preslikavo pri koncnih parametrih        
    oMat2D = transAffine2D( iTrans=oPar )        
    # preslikaj premicno sliko        
    oImgReg = transformImage( iImgMov, oMat2D ).astype( iImgType )
    # vrni rezultat
    return oPar, oImgReg

# test funkcije
if __name__ == '__main__':
    # doloci fiksno in premicno sliko
    oPar = [0, 1]
    iImgFix = oVideo[:,:,0]
    iImgMov = transformImage( iImgFix, transAffine2D( iTrans = oPar ) )
    # klici Lucas-Kanade poravnavo slik
    import time    
    ts = time.clock()    
    oPar, oImgReg = regLucasKanade( iImgFix, iImgMov, 20 )
    print('parameters: ', oPar)
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    showImage( iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo' )
    showImage( iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi' )

#%% Naloga 3: Funkcija za poravnavo s piramidnim Lucas-Kanade postopkom
def regPyramidLK( iImgFix, iImgMov, iMaxIter, iNumScales, iVerbose=True ):
    """Piramidna implementacija poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgFix = np.array( iImgFix, dtype='float' )
    iImgMov = np.array( iImgMov, dtype='float' )
    # pripravi piramido slik
    iPyramid = [ (iImgFix, iImgMov) ]
    for i in range(1,iNumScales):
        # decimiraj fiksno in premicno sliko za faktor 2
        iImgFix_2 = decimateImage2D( iImgFix, i )
        iImgMov_2 = decimateImage2D( iImgMov, i )
        # dodaj v seznam
        iPyramid.append( (iImgFix_2,iImgMov_2) )
    # doloci zacetne parametre preslikave
    oPar = np.array( (0,0) )          
    # izvedi poravnavo od najmanjse do najvecje locljivosti slik
    for i in range(len(iPyramid)-1,-1,-1):
        if iVerbose: print('PORAVNAVA Z DECIMACIJO x%d' % 2*i)
        # posodobi parametre preslikave
        oPar = oPar * 2.0
        # izvedi poravnavo pri trenutni locljivosti
        oPar, oImgReg = regLucasKanade( iPyramid[i][0], iPyramid[i][1], \
                            iMaxIter, oPar, iVerbose=iVerbose )
    # vrni koncne parametre in poravnano sliko
    return oPar, oImgReg

# test funkcije
if __name__ == '__main__':
    # doloci fiksno in premicno sliko
    oPar = [0, 10]
    iImgFix = oVideo[:,:,0]
    iImgMov = transformImage( iImgFix, transAffine2D( iTrans = oPar ) )
    # klici Lucas-Kanade poravnavo slik
    import time    
    ts = time.clock()    
    oPar, oImgReg = regPyramidLK( iImgFix, iImgMov, 20, 3 )
    print('parameters: ', oPar)
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    showImage( iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo' )
    showImage( iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi' )
    
#%% Naloga 4: Funkcija za sledenje tarci z Lucas-Kanade postopkom
def trackTargetLK( iVideoMat, iCenterXY, iFrameXY, iVerbose=True ):
    """Postopek sledenja Lucas-Kanade"""
    # pretvori vhodni video v numpy polje
    iVideoMat = np.asarray( iVideoMat )
    iCenterXY = np.array( iCenterXY )
    # definiraj izhodno spremenljivko
    oPathXY = np.array( iCenterXY.flatten() ).reshape( (1,2) )
    # definiraj koordinate v tarci
    gx, gy = np.meshgrid( range(iFrameXY[0]), range(iFrameXY[1]) )
    gx = gx - float(iFrameXY[0]-1)/2.0
    gy = gy - float(iFrameXY[1]-1)/2.0
    # zazeni LK preko vseh zaporednih okvirjev
    for i in range(1,iVideoMat.shape[-1]):
#        print('PORAVNAVA OKVIRJEV %d-%d' % (i-1,i) )
        # vzorcni tarco v dveh zaporednih okvirjih        
        iImgFix = interpolate1Image2D( iVideoMat[...,i-1], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1] )
        iImgMov = interpolate1Image2D( iVideoMat[...,i], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1] )
        # zazeni piramidno LK poravnavo
        oPar, oImgReg = regPyramidLK( iImgFix, iImgMov, 30, 3, iVerbose=False )
        # shrani koordinate
        oPathXY = np.vstack( (oPathXY, oPathXY[-1,:] + oPar.flatten()) )     
        print('koordinate tarce: ', oPathXY[-1,:])
    # vrni spremenljivko
    return oPathXY

# test funkcije
if __name__ == '__main__':
    # klici Lucas-Kanade sledenje tarci
    import time    
    ts = time.clock()    
    oPathXY = trackTargetLK( oVideo[...,:], (33,370), (40,40) )
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    # prikazi tarco in pot v razlicnih okvirjih 
    drawPathToFrame( oVideo, oPathXY, iFrame=1, iFrameSize=(40,40),  )
    drawPathToFrame( oVideo, oPathXY, iFrame=100, iFrameSize=(40,40) )
    drawPathToFrame( oVideo, oPathXY, iFrame=170, iFrameSize=(40,40) )   


def cross2matrix(x):
    '''
    Antisimetricna matrika na podlagi trivrsticnega vektorja

    Vrne antisimetricno matriko M, ki ustreza 3-vektorju x tako, da
    velja M*y = cross(x,y) za vse 3-vektorje y.

    Parameters
    ----------
    x : numpy.ndarray
        Vhodni 3-vektor

    Returns
    ---------
    oMat : numpy.ndarray
           3x3 matrika
    '''
    return np.array(
        (
            (0, -x[2], x[1]),
            (x[2], 0, -x[0]),
            (-x[1], x[0], 0)
        ))


def linearTriangulation(x1, x2, P1, P2):
    '''
    Linearna triangulacija

    Parameters:
    ----------
    x1 : numpy.ndarray
         Homogene koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene koordinate točk v sliki 2 (3xN)
    P1 : numpy.ndarray
         Projekcijska matrika kamere 1 (3x4)
    P2 : numpy.ndarray
         Projekcijska matrika kamere 2 (3x4)

    Returns
    ---------
    X : numpy.ndarray
        Homogene koordinate 3D točk (4xN)
    '''
    # preveri dimenzije polj
    assert x1.shape[0] == x2.shape[0], 'Size mismatch of input points'
    assert x1.shape[1] == x2.shape[1], 'Size mismatch of input points'
    assert x1.shape[0] == 3, 'Arguments x1, x2 should be 3xN matrices (homogeneous coords)'
    assert P1.shape == (3, 4), 'Projection matrices should be of size 3x4'
    assert P2.shape == (3, 4), 'Projection matrices should be of size 3x4'

    num_points = x1.shape[1]

    P = np.zeros((4,num_points))

    # linearni algoritem
    for j in range(num_points):
        # sestavi matriko linearnega homogenega sistema enacb
        A1 = np.dot(cross2matrix(x1[:, j]), P1)
        A2 = np.dot(cross2matrix(x2[:, j]), P2)
        A = np.vstack((A1, A2))

        # resi linearni homogeni sistem enacb
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        P[:,j] = vt[-1, :]

    # dehomogeniziraj (P je izrazen v homogenih koordinatah)

    return P / P[-1,:]

def normalise2dpts(pts):
    '''
    Normalizacija 2D homogenih tock

    Funkcija normalizira vsak set tock tako, da bo izhodisce v centroidu in
    povprecna razdalja do tock od izhodisca sqrt(2)

    Parameters:
    ----------
    pts : numpy.ndarray
         Polje homogenih koordinat 2D točk v sliki 1 (3xN)

    Returns
    ---------
    new_pts : numpy.ndarray
        Polje preslikanih homogenih koordinat 2D točk v sliki 1 (3xN)
    T : numpy.ndarray
        Transformacijska matrika 3x3: newpts = T * pts
    '''
    num_points = pts.shape[-1]

    # centroid
    mu = np.mean(pts[:2,:], axis=-1).reshape((2,1))
    # povprecna standardna deviacija
    sigma = np.mean(np.sqrt(np.sum((pts[:2,:] - mu)**2.0, axis=0)))
    s = np.sqrt(2) / sigma

    T = np.array(
        (
            (s, 0, -s * mu[0]),
            (0, s, -s * mu[1]),
            (0, 0, 1)
         )
    )

    return np.dot(T, pts), T

def fundamentalEightPoint(x1, x2):
    '''
    Algoritem 8-tock za oceno fundamentalne matrike F

    Rang 3x3 matrike F je v splosnem enak 2, kar ta funkcija zagotavlja preko
    nastavljanja singularnih vrednosti SVD razcepa. Funkcija ne vkljucuje normalizacije.

    Referenca: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

    Parameters:
    ----------
    x1 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 2 (3xN)

    Returns
    ---------
    F : numpy.ndarray
        Fundamentalna matrika (3x3)
    '''
    # preveri dimenzije vhodnih spremenljivk
    assert x1.shape == x2.shape, 'Size mismatch of input points'
    assert x1.shape[0], 'Input arguments are not 2D points'
    assert x1.shape[-1] >= 8, 'Insufficient number of points to compute fundamental matrix (need >=8)'

    num_points = x1.shape[-1]

    # izracunaj matriko A linearnega homogenega sistema, resitev katere je vektor
    # z elementi fundamentalne matrike
    A = np.zeros((num_points, 9))
    for i in range(num_points):
        A[i, :] = np.kron(x1[:, i], x2[:, i]).T

    # resi linearni homogeni sistem enacb kot A*f=0
    # korespondence x1<->x2 so eksaktne == rank(A)=8, torej resitev obstaja
    # ce so meritve koordinat x1, x2 obremenjene s sumom, potem je rank(A)=9, torej ne obstaja
    # eksaktna resitev in moramo poiskati resitev z minimizacijo srednje kvadratne napake
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F = np.reshape(Vt[-1, :], (3, 3)).T

    # vsili det(F)=0 s projekcijo ocenjene matrike F na mnozico 3x3 singularnih matrik
    u, s, vt = np.linalg.svd(F)
    s[2] = 0
    return np.dot(u, np.dot(np.diag(s), vt))

def fundamentalEightPoint_normalized(x1, x2):
    '''
    Oceni fundamentalno matriko glede na dane pripadajoce pare tock in
    projekcijsko matriko kamere K (intrinzicni parametri)

    Parameters:
    ----------
    x1 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 2 (3xN)

    Returns
    ---------
    F : numpy.ndarray
        Fundamentalna matrika (3x3)
    '''
    # normaliziraj vsak set tock tako, da bo izhodisce v centroidu in
    # povprecna razdalja do tock od izhodisca sqrt(2)
    x1_nh, T1 = normalise2dpts(x1)
    x2_nh, T2 = normalise2dpts(x2)

    # linearna resitev
    F = fundamentalEightPoint(x1_nh, x2_nh)

    # kompenzacija normalizacije
    return np.dot(T2.T, np.dot(F, T1))

def estimateEssentialMatrix(x1, x2, K1, K2):
    '''
    Oceni esencialno matriko glede na dane pripadajoce pare tock in
    projekcijsko matriko kamere K (intrinzicni parametri)

    Parameters:
    ----------
    x1 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 2 (3xN)
    K1 : numpy.ndarray
         Projekcijska matrika kamere 1 (3x4)
    K2 : numpy.ndarray
         Projekcijska matrika kamere 2 (3x4)

    Returns
    ---------
    E : numpy.ndarray
        Esencialna matrika (3x3)
    '''
    # oceni fundamendalno matriko K2^-T * F * K1^-1
    F = fundamentalEightPoint_normalized(x1, x2)

    # izracunaj esencialno matriko iz ocenjene fundamentalne matrikeCompute the essential matrix from the fundamental matrix given K
    return np.dot(K2.T, np.dot(F, K1))

def distPoint2EpipolarLine(F, x1, x2):
    '''
    Izracunaj razdaljo med tocko in epipolarno crto

    Parameters:
    ----------
    F : numpy.ndarray
        Fundamentalna matrika
    x1 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene 2D koordinate točk v sliki 2 (3xN)

    Returns
    ---------
    dist2 : float
        vsota kvadriranih razdalj od tock do epipolarne crte, normalizirana s stevilom koordinat tock
    '''
    num_points = x1.shape[-1]

    homog_points = np.hstack((x1, x2))
    epi_lines = np.hstack((np.dot(F.T, x2), np.dot(F, x1)))

    denom = epi_lines[0,:]**2.0 + epi_lines[1,:]**2.0
    return np.sqrt(
        np.sum(
            (np.sum(epi_lines*homog_points, axis=0)**2.0) / denom
        ) / num_points
    )

def decomposeEssentialMatrix(E):
    '''
    Za dano esencialno matriko izracunaj gibanje kamere, tj. R in T tako, da bo E ~ T_x R
    function [R,u3] = decomposeEssentialMatrix(E)

    Parameters:
    ----------
    E : numpy.ndarray
        Esencialna matrika (3x3)

    Returns
    ---------
    R : numpy.array
        Polje (3x3x2) dveh moznih rotacij
    u3 : numpy.array
        Vektor s translacijo'''
    U, zl, Vt = np.linalg.svd(E)

    #translacija
    u3 = U[:,2]

    # rotacija
    R = np.zeros((3,3,2))
    W = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    R[:,:,0] = np.dot(U, np.dot(W, Vt))
    R[:,:,1] = np.dot(U, np.dot(W.T, Vt))

    if np.linalg.det(R[:,:,0])<0:
        R[:,:,0] = -R[:,:,0]

    if np.linalg.det(R[:,:,1])<0:
        R[:,:,1] = -R[:,:,1]

    if np.linalg.norm(u3) != 0:
        u3 = u3 / np.linalg.norm(u3)

    return R, u3

def disambiguateRelativePose(Rots, u3, points0_h, points1_h, K0, K1):
    '''
    Poisci pravilno relativno pozo kamere (med stirimi moznimi) in vrni tisto, pri
    kateri tocke lezijo pred slikovno ravnino (s pozitivno globino)

    Parameters:
    ----------
    Rots : numpy.array
        Polje (3x3x2) dveh moznih rotacij
    u3 : numpy.array
        Vektor s translacijo
    x1 : numpy.ndarray
         Homogene koordinate točk v sliki 1 (3xN)
    x2 : numpy.ndarray
         Homogene koordinate točk v sliki 2 (3xN)
    K1 : numpy.ndarray
         Kalibracijska matrika kamere 1 (3x3)
    K2 : numpy.ndarray
         Kalibracijska matrika kamere 2 (3x3)

    Returns
    ---------
    R : numpy.array
        Polje (3x3) rotacije
    T : numpy.array
        Vektor s translacijo

        kjer [R|t] = T_C1_C0 = T_C1_W predstavlja transformacijo tock iz
        svetovnega koordinatnega sistema (enak tistemu od kamere 1) v kamero 2
    '''
    # projekcijska matrika kamere 1
    M0 = np.dot(K0, np.eye(3,4))

    total_points_in_front_best = 0
    for iRot in (0,1):
        R_C1_C0_test = Rots[:, :, iRot]

        for iSignT in (1,2):
            T_C1_C0_test = (u3 * (-1)**iSignT).reshape((3, 1))

            M1 = np.dot(K1, np.hstack((R_C1_C0_test, T_C1_C0_test)))
                        
            P_C0 = linearTriangulation(points0_h,points1_h, M0, M1)

            # projekcija v obe kameri
            P_C1 = np.dot(np.hstack((R_C1_C0_test, T_C1_C0_test)), P_C0)

            num_points_in_front0 = np.sum(P_C0[2, :] > 0)
            num_points_in_front1 = np.sum(P_C1[2, :] > 0)
            total_points_in_front = num_points_in_front0 + num_points_in_front1

            if (total_points_in_front > total_points_in_front_best):
                # shrani rotacijo, ki vrne najvecje stevilo tock pred obema kamerama
                R = R_C1_C0_test
                T = T_C1_C0_test
                total_points_in_front_best = total_points_in_front

    return R, T

def mapAffineApprox2D(iPtsRef, iPtsMov):
    """Afina aproksimacijska poravnava Zanima nas kaksna je najboljsa aproksimacija da te tocke poravnamo ene v druge"""
    # YOUR CODE HERE
    #Begin SOlution
    iPtsRef = np.matrix(iPtsRef)
    iPtsMov = np.matrix(iPtsMov)
    #po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    #Afina aproksimacija s psevdoinverzom
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    #psevdoinvezr
    oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))
    #psevdoinverz na dolgo in siroko:
    
    #oMat2D = iPtsRef * iPtsMov.transpose() * \
    #np.linalg.inv(iPtsMov * iPtsMov.transpose() )
    #END
    
    #ZMER RABS 3x3 MATRIKO DRGAC SI U KURCU
    
    
    return oMat2D

def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # YOUR CODE HERE - ICP preslikava
    #inicializiraj izhodne parametre
    curMat = []; oErr = []; iCurIter = 0
    if plotProgress: #plotProgress je da se izrisuje po korakih
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
    #Zacni iterativni postopek
    while True:
        #poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        
        #doloci afino aprkosimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        
        #Posodobi premicne tocke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        
        #izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2]-iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1
        
        #preveri kontrolne parametre
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter > iMaxIter or np.all(dMat < iEps): #ce smo presegli stevilo dovoljenih iteracij ali ce so vsi elementi te matrike vecji od iEps bomo izstopl iz te iteracije
            break
            
    #doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
                
        if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()
            ax.plot(iPtsRef[:,0], iPtsRef[:,1], 'ob')
            ax.plot(iPtsMov_t[:,0], iPtsMov_t[:, 1], 'om')
            fig.canvas.draw()
            plt.pause(1)
            
        oMat2D = np.dot(curMat[i], oMat2D)
        
                    

    return oMat2D, oErr

def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # YOUR CODE HERE
    #inicializiraj polje indeksov
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)
    
    idxPair = -np.ones ((iPtsRef.shape[0], 1), dtype = 'int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i, j] = np.sum((iPtsRef[i, :2] - iPtsMov[j, :2])**2) #razdalja obeh vektorjev pomnozenih emd seboj
            
    #Doloci bijektivno preslikavo
    while not np.all(idxDist == np.inf): #to delamo dokler ne pokrijemo vseh elementov v idxDist matriki
        i,j = np.where(idxDist == np.min(idxDist)) # s funkcijo where pogledamo na katerem indeksu je razdalja med dvema tockama najmanjsa
        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf
        idxDist[:, j[0]] = np.inf
        
    #doloci pare tock
    idxValid, idxNotValid = np.where(idxPair >= 0)
    idxValid = np.array(idxValid)
    iPtsRef_t = iPtsRef[idxValid, :]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(), :]

    return iPtsRef_t, iPtsMov_t

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    #assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])