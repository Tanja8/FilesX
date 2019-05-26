# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:16:43 2015

@author: Žiga Špiclin

RVLIB: knjižnica funkcij iz laboratorijskih vaj
       pri predmetu Robotski vid
"""

import scipy.ndimage as ni
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp
from scipy.interpolate import interpn
import matplotlib.animation as animation


def showImage(iImage, iTitle=''):
    '''
    Prikaže sliko iImage in jo naslovi z iTitle
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    
    Returns
    ---------
    Nothing
    
    
    '''
    plt.figure() # odpri novo prikazno okno
    
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])

    plt.imshow(iImage, cmap = cm.Greys_r) # prikazi sliko v novem oknu
    plt.suptitle(iTitle) # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')

    
def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika
        
    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0,:,:].astype('float')
    g = iImage[1,:,:].astype('float')
    b = iImage[2,:,:].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)
   
    
def discreteConvolution2D(iImage, iKernel):
    '''
    Diskretna 2D konvolucija slike s poljubnim jedrom

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iKernel : numpy.ndarray
        Jedro ali matrika za konvolucijo
        
    Returns
    --------
    oImage : numpy.ndarray
        Z jedrom konvolirana vhodna slika
    '''
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray(iImage)
    iKernel = np.asarray(iKernel)
    #------------------------------- za hitrost delovanja
    oImage = ni.convolve(iImage, iKernel, mode='nearest')    
    return oImage

    
def imageGradient(iImage):
    '''
    Gradient slike s Sobelovim operatorjem

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika
        
    Returns
    --------
    oImage : tuple of two numpy.ndarray variables
        S Sobelovim jedrom konvolirana vhodna slika
    '''    
    iImage = np.array(iImage, dtype='float')    
    iSobel = np.array(((-1,0,1),(-2,0,2),(-1,0,1)))    
    oGx = ni.convolve(iImage, iSobel, mode='nearest')
    oGy = ni.convolve(iImage, np.transpose(iSobel), mode='nearest')
    return oGx, oGy

    
def decimateImage2D(iImage, iLevel):
    '''
    Funkcija za piramidno decimacijo

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iLevel : int
        Število decimacij s faktorjem 2
        
    Returns
    --------
    oImage : numpy.ndarray
        Decimirana slika
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray(iImage)
    iImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array(((1/16,1/8,1/16),(1/8,1/4,1/8),(1/16,1/8,1/16)))
    # glajenje slike pred decimacijo
    iImage = discreteConvolution2D(iImage, iKernel)
    # decimacija s faktorjem 2
    iImage = iImage[::2,::2]
    # vrni sliko oz. nadaljuj po piramidi
    if iLevel <= 1:
        return np.array(iImage, dtype=iImageType)
    else:
        return decimateImage2D(iImage, iLevel-1)   
      
        
def interpolate1Image2D(iImage, iCoorX, iCoorY):
    '''
    Funkcija za interpolacijo prvega reda

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iCoorX : numpy.ndarray
        Polje X koordinat za interpolacijo

    iCoorY : numpy.ndarray
        Polje Y koordinat za interpolacijo
        
    Returns
    --------
    oImage : numpy.ndarray
        Interpolirane vrednosti v vhodnih koordinatah X in Y
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray(iImage)    
    iCoorX = np.asarray(iCoorX)
    iCoorY = np.asarray(iCoorY)   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')
    #------------------------------- za hitrost delovanja    
    return interpn((np.arange(dy),np.arange(dx)), iImage, \
                      np.dstack((iCoorY,iCoorX)),\
                      method='linear', bounds_error=False)\
                      .astype(iImage.dtype)    


def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    '''
    Ustvari poljubno 2D afino preslikavo v obliki 3x3 homogene matrike

    Parameters
    ----------
    iScale : tuple, list
        Skaliranje vzdolž x in y

    iTrans : tuple, list
        Translacija vzdolž x in y

    iRot : float
        Kot rotacije

    iShear : tuple, list
        Strig vzdolž x in y
        
    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika

    '''  
    iRot = iRot * np.pi / 180
    oMatScale = np.matrix(((iScale[0],0,0),(0,iScale[1],0),(0,0,1)))
    oMatTrans = np.matrix(((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)))
    oMatRot = np.matrix(((np.cos(iRot),-np.sin(iRot),0),\
                          (np.sin(iRot),np.cos(iRot),0),(0,0,1)))
    oMatShear = np.matrix(((1,iShear[0],0),(iShear[1],1,0),(0,0,1)))
    # ustvari izhodno matriko
    oMat2D = oMatTrans * oMatShear * oMatRot * oMatScale
    return oMat2D               
               
    
def transformImage(iImage, oMat2D):
    '''
    Preslikava 2D slike z linearno preslikavo

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
        
    Returns
    --------
    oImage : numpy.ndarray
        Izhodna preslikana sivinska slika

    '''
    # ustvari diskretno mrezo tock
    gx, gy = np.meshgrid(range(iImage.shape[1]), \
                          range(iImage.shape[0]), \
                          indexing = 'xy')    
    # ustvari Nx3 matriko vzorcnih tock                          
    pts = np.vstack((gx.flatten(), gy.flatten(), np.ones((gx.size,)))).transpose()
    # preslikaj vzorcne tocke
    pts = np.dot(pts, oMat2D.transpose())
    # ustvari novo sliko z interpolacijo sivinskih vrednosti
    oImage = interpolate1Image2D(iImage, \
                                  pts[:,0].reshape(gx.shape), \
                                  pts[:,1].reshape(gx.shape))
    oImage[np.isnan(oImage)] = 0
    return oImage 
    
    
def showVideo(iVideo, oPathXY=np.array([])):
    '''
    Prikaz video animacije poti

    Parameters
    ----------
    iVideo : numpy.ndarray
        Vhodni sivinski video v 3D (NxMxt) polju

    oPathXY : numpy.ndarray
        Zaporedje (x,y) koordinat poti objekta
        
    Returns
    --------
    

    '''
    global oVideo_t, iFrame, oPathXY_t
    # funkcija za osvezevanje prikaza    
    def updatefig(*args):
        '''
        Pomozna funkcija za prikaz videa
    
        '''    
        global oVideo_t, iFrame, oPathXY_t
        iFrame = (iFrame + 1) % oVideo_t.shape[-1]
        im.set_array(oVideo_t[...,iFrame]) 
        if iFrame < oPathXY.shape[0]:
            plt.plot(oPathXY[iFrame,0], oPathXY[iFrame,1], 'xr' ,markersize=3)    
        return im,    
    fig = plt.figure()
    plt.ioff
    # prikazi prvi okvir
    iFrame = 0
    oPathXY_t = oPathXY
    oVideo_t = iVideo
    im = plt.imshow(iVideo[...,iFrame], cmap=plt.get_cmap('Greys_r'))
    # prikazi animacijo poti
    ani = animation.FuncAnimation(fig, updatefig, interval=25, blit=True)
    plt.show()
    return ani

    
def drawPathToFrame(oVideo, oPathXY, iFrame=1, iFrameSize=(40,40)):
    '''
    Prikaz poti do izbranega okvirja

    Parameters
    ----------
    iVideo : numpy.ndarray
        Vhodni sivinski video v 3D (NxMxt) polju

    oPathXY : numpy.ndarray
        Zaporedje (x,y) koordinat poti objekta
        
    Returns
    --------
    

    '''
    oPathXY_t = oPathXY[:iFrame,:]
    showImage(oVideo[...,iFrame], 'Pot do okvirja %d' % iFrame)
    for i in range(1,oPathXY_t.shape[0]):
        plt.plot(oPathXY_t[i-1:i+1,0],oPathXY_t[i-1:i+1,1],'--r')
        if i==1 or (i%5)==0:
            plt.plot(oPathXY_t[i,0],oPathXY_t[i,1],'xr',markersize=3)
        
    dx = iFrameSize[0]/2; dy = iFrameSize[1]/2
    plt.plot((oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]+dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot((oPathXY_t[-1,0]+dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot((oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]-dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')
    plt.plot((oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]-dy),'-g')


def loadVideo( iFileName, iFrameSize = (576, 720) ):
    '''
    Naloganje videa z ffmpeg orodjem
    
    Za nalaganje videa boste potrebovali knjiznico ffmpeg (datoteko ffmpeg.exe),
    ki jo lahko nalozite s spletne strani https://www.ffmpeg.org/download.html
    
     ----------
    iFileName : str 
        Pot do mape in ime datoteke

    iFrameSize : tuple
        NxM dimenzije posameznega okvirja v videu
        
    Returns
    --------   
    iVideo : numpy.ndarray
        Izhodni sivinski video v 3D (NxMxt) polju
        
    '''
    import sys
    import subprocess as sp
    # ustvari klic ffmpeg in preusmeri izhod v cevovod
    command = [ 'ffmpeg.exe',
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
        sys.stdout.write("\rBerem okvir %d ..." % frameCount)
        sys.stdout.flush()
        # preberi Y*X*3 bajtov (= 1 okvir)
        raw_frame = pipe.stdout.read(np.prod(iFrameSize)*3)
        # pretvori prebrane podatke v numpy polje
        frame =  np.fromstring(raw_frame, dtype='uint8')       
        # preveri ce je velikost ustrezna, sicer prekini zanko  
        if frame.size != (np.prod(iFrameSize)*3):
            sys.stdout.write(" koncano!\n")
            break;
        # preoblikuj dimenzije in pretvori v sivinsko sliko
        frame = colorToGray( frame.reshape((iFrameSize[0], iFrameSize[1], 3)).
                                            transpose([2,0,1]))
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