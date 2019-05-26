import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import time
from skimage.measure import compare_ssim
import argparse
import imutils
import time    
from scipy.spatial import distance as dist
from ipywidgets import interact, IntSlider
from funkcije import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import PIL.Image as im
import socket
import struct
import glob




######################################            library              ##################################3

def captureImage(photoName):
    """Zajem slike objekta"""
    camera = cv2.VideoCapture(0)
    time.sleep(2)
    return_value, image = camera.read()
    cv2.imwrite(photoName, image)
    del(camera)


def prepareImage(img1,backGr1,img2,backGr2):    
    
    #ustavarimo sivinske slike
    grayA1 = cv2.cvtColor(imageA1, cv2.COLOR_BGR2GRAY)
    grayB1 = cv2.cvtColor(imageB1, cv2.COLOR_BGR2GRAY)

    grayA2 = cv2.cvtColor(imageA2, cv2.COLOR_BGR2GRAY)
    grayB2 = cv2.cvtColor(imageB2, cv2.COLOR_BGR2GRAY)

    #izracun razlik med slikama
    (score1, diff1) = compare_ssim(grayA1, grayB1, full=True)
    diff1 = (diff1 * 255).astype("uint8")

    (score2, diff2) = compare_ssim(grayA2, grayB2, full=True)
    diff2 = (diff2 * 255).astype("uint8")

    #upragovljanje slike razlik, iskanje kontur
    thresh1 = cv2.threshold(diff1, 0, 255,
	    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 

    #upragovljanje slike razlik, iskanje kontur
    thresh2 = cv2.threshold(diff2, 0, 255,
	    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 

    # izlocm nepravilnosti v svetlobi - uporabimo morfološko operacijo odpiranja
    kernel = np.ones((5,5),np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5,5),np.uint8)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
    return grayA1,grayA2,diff1,diff2,thresh1,thresh2

def findFruit(thresh1, imageA, printOut):
    """ thresh1 - upragovljena slika razlik"""
    """ imageA - slika na katero želimo narisati zaznani sadež"""

    #poiscemo konture samo na prvi upragovljeni sliki razlik
    cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #objekt zanimanje je le največja kontura
    #numb_of_obj = len(cnts)
    numb_of_obj = 1

    #sortiranje kontur po velikosti porvšine - največja [-1]
    largest_contours = sorted(cnts, key=cv2.contourArea)[-numb_of_obj:]

    # loop over the contours
    for c in largest_contours:
        #narisemo pravokotnike na centre najdenih kontur
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
       
 
    axesL = []
    #izracun povrsine in obsega vseh nubm_of_obj kontur
    for i in range(0,numb_of_obj):

        #izracun centra sadeža
        M = cv2.moments(largest_contours[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        #narišemo center sadeža
        cv2.circle(imageA,(cx,cy), 5, (0,0,255), -1)

        (x, y, w, h) = cv2.boundingRect(largest_contours[i])
    
        #fitanje najmanješga kroga, ki objame i. konturo
        (x,y),radius = cv2.minEnclosingCircle(largest_contours[i])
        center = (int(x),int(y))
        radius = int(radius)
        #cv2.circle(imageB1,center,radius,(255,0,0),2)

        #pretvorba v hsv prostor, ocena barve centra sadeža
        #hsv = cv2.cvtColor(imageA_HSV,cv2.COLOR_BGR2HSV)
        #print("hsv vrednosti:",hsv[[cx],[cy]]) # hsv vrednosti centra sadeža

        #izracun lastnosti konture
        area = cv2.contourArea(largest_contours[i])
        perimeter = cv2.arcLength(largest_contours[i],True)
        areaVSp = area/perimeter/radius*2
        axesL = w/h

        if printOut :
            print("-------------------------------------------------------------------------------------------------------------------------")
            print("Površina",i,".največje konture:",area)
            print("Obseg",i,".največje konture:",perimeter)
            print("Povrsina/obseg * 2/radius",i,". konture =", areaVSp)
            print("koordinate centra",i,". konture:",cx,cy)
            print("razmerje stranic pravokotnika, ki objame sadež",i,". konture:",axesL)
            print("-------------------------------------------------------------------------------------------------------------------------")
        return cx,cy,w,h,areaVSp,axesL,imageA,largest_contours

def decideFruitType(areaVSp,axesL):
    #### ODLOČANJE KATERI SADEŽ IMAMO
    class Fruit:
     def __init__(self,name,shape, axisL,color,wh):
        self.name = name    
        self.shape = shape
        self.axisL = axisL # najdaljša os sadeža
        self.color = color
        self.wh = wh

    # v zbirki za rezanje imamo trnutno le 2 sadeža
    p = []
    p1 = Fruit("Banana",0.3, 200, range(0,70),0.3)
    p2 = Fruit("Apple",0.9, 100, range(60,120),1)
    p3 = Fruit("X",areaVSp, 100, 0 ,axesL)
    p = [p1,p2,p3]

    fDiff_shape = []
    fDiff_axisL = []

    for i in range(0,len(p)-1):
        fDiff_shape.append(abs(p[i].shape - p[-1].shape))
        fDiff_axisL.append(abs(p[i].axisL - p[-1].axisL))


    minVal =np.min(fDiff_shape)
    indexO = fDiff_shape.index(minVal)  
    print("Sadež na sliki je:",p[indexO].name)
    return indexO

def getFruitPOints(diff1,diff2):
    
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.45

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(diff1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(diff2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
   
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
 
    # Draw top matches
    imMatches = cv2.drawMatches(diff1, keypoints1, diff2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros(((len(matches)+1), 2), dtype=np.float32)
    points2 = np.zeros(((len(matches)+1), 2), dtype=np.float32)
   

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    #pogoj da so 1. točke znotraj konture
    iImgFix1 = np.array(largest_contours).astype('float')
    points1C = np.zeros((len(iImgFix1[0,:,0,0]), 2), dtype=np.float32)


    for i in range(0,len(iImgFix1[0,:,0,0])):
        points1C[i] = iImgFix1[0,i,0,:]

    points1 = np.array(points1)
    points1C = np.array(points1C)
    points1.astype(int)
    points1C.astype(int)
    dist = []
    dist = np.array(dist,dtype=bool)

    lastOne = (len(points1) -1)

    for i in range(0,lastOne):
        dist1= cv2.pointPolygonTest(points1C,(points1[i,0],points1[i,1]),True)
        dist2= cv2.pointPolygonTest(points1C,(points2[i,0],points2[i,1]),True)

        if dist1 < 0 :
            points1[i,:] = points1[i-1,:] # če točka ni znotraj konture,prepišemo prejšno (majhna vrjetnos, da je tudi ta izven)
            points2[i,:] = points2[i-1,:]



    # doloci matriko preslikave med slikama
    #oMat2D, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    #kot zadnjo točko dodam center sadeža določen 2D, ki ga kasneje preslikamo v 3D
    #points1[len(matches), :] = [cx,cy]

    #preslikaj koordinate prve slike za tranformacjo med slikama
    #points2 = np.dot(addHomCoord2D(points1), oMat2D.transpose())
 
    #zlozim array
    O = np.array([points1[:,0],points1[:,1]])
    O_t = np.array([points2[:,0],points2[:,1]])

    #shranim oba para točk
    np.savetxt('points1.txt', O)
    np.savetxt('points2.txt', O_t)
    return len(matches)


def findFruit3Dpose(NumbOfCent):
    # K matrika naše kamere, predhodno smo izvedli kalibracijo
    K= np.genfromtxt('CameraK.txt')
 
    p1 = np.genfromtxt('points1.txt')
    p2 = np.genfromtxt('points2.txt')

    #faktor preslikave med relativnimi in geometrijskimi kooordinatami
    geomFakt = 110

    p1 = np.vstack((p1, np.ones((1, p1.shape[-1]))))
    p2 = np.vstack((p2, np.ones((1, p2.shape[-1]))))

    # oceni esencialno matriko z algoritmom 8-tock
    E = estimateEssentialMatrix(p1, p2, K, K)
    E = E.astype(float)

    # izloci relativen polozaj kamere (R, T) iz esencialne matrike
    Rots, u3 = decomposeEssentialMatrix(E)

    # najdi resitev med 4-imi moznimi
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

    R_C2_W = R_C2_W
    T_C2_W = T_C2_W*geomFakt


    print("položaj kamere glede na prvo",T_C2_W)

    # izvedi triangulacijo oblaka tock s koncno preslikavo (R, T)
    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot(K, np.hstack((R_C2_W, T_C2_W)))
    P = linearTriangulation(p1, p2, M1, M2)

    P = P* geomFakt# RAZMERJE MED RELATIVNIMI ENOTAMI V SLIKI IN PRAVIMI METRIČNIMI - POZNAMO IZ ZNANEGA PREMIKA KAMERE

    centerPX = []
    centerPY = []
    centerPZ = []

    #centerPX = P[0, NumbOfCent] # zadnja točka je center
    #centerPY = P[1, NumbOfCent]
    #centerPZ = P[2, NumbOfCent]

    centerPX = np.mean(P[0, :])
    centerPY = np.mean(P[1, :])
    centerPZ = np.mean(P[2, :])

    centerP = []
    centerP = [centerPX, centerPY, centerPZ]

    print(" Center sadeža konponenta X",centerPX," Center sadeža konponenta Y", centerPY, "Center sadeža konponenta Z",centerPZ)
    return centerP


def send_vals(rot1,rot2,rot3,rot4,rot5,rot6,rot7,rot8,rot9,tr1,tr2,tr3):

    RAD_REAL = 0.15
    ref_center = None
    ratioCoeff = None
    
    client_ip = "192.168.65.40"
    port = 25150
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP 
    vals = (rot1,rot2,rot3,rot4,rot5,rot6,rot7,rot8,rot9,tr1,tr2,tr3)

    packer = struct.Struct('f f f f f f f f f f f f')
    bin_data = packer.pack(*vals)
    sock.sendto(bin_data, (client_ip, port)) 



def GetFruitOrientation(largest_contours):
    #referenčna slika za orientacijo, poravana na naš gib za rezanje
    imageARef = cv2.imread("Fruit0Rotation.jpg")
    imageBRef = cv2.imread("Fruit0RotationBG.jpg")

    _ ,_,_,_,thresh1,_ = prepareImage(imageARef,imageBRef,imageA2,imageB2)
    _,_,_,_,_,_,_,ref_contours = findFruit(thresh1, imageARef,False)

    iImgFix2 = np.array(largest_contours).astype('float')
    iImgFix1 = np.array(ref_contours).astype('float')

    ContourSize = []
    ContourSize.append(len(iImgFix1[0,:,0,0]))
    ContourSize.append(len(iImgFix2[0,:,0,0]))
    CS = np.min(ContourSize)

    points1 = np.zeros((CS, 2), dtype=np.float32)
    points2 = np.zeros((CS, 2), dtype=np.float32)

    for i in range(0,CS):
        points1[i] = iImgFix1[0,i,0,:]
        points2[i] = iImgFix2[0,i,0,:]

    RotMat2D = mapAffineApprox2D(points1, points2)
    Rot = rotationMatrixToEulerAngles(RotMat2D)
    return Rot


################################################## MAIN ###################################################################

""" ##########################################  Zajem slik #################################################### """
#zapelji robota v osnovno lego za slikanje, slika brez sadeža, ko je na mestu, pritisnite katerokoli tipko za nadaljevanje
cv2.waitKey(0)
captureImage('test1.jpg')

#zapelji robota v osnovno lego za slikanje, slika sadeža, ko je na mestu, pritisnite katerokoli tipko za nadaljevanje
cv2.waitKey(0)
captureImage('test2.jpg')

#zapelji robota v zamaknjeno lego za slikanje, slika sadeža, ko je na mestu, pritisnite katerokoli tipko za nadaljevanje
cv2.waitKey(0)
captureImage('test11.jpg')

#zapelji robota v zamaknjeno lego za slikanje, slika brez sadeža, ko je na mestu, pritisnite katerokoli tipko za nadaljevanje
cv2.waitKey(0)
captureImage('test22.jpg')




""" Naložite zajete slike, ki jih želite uporabiti za obdelavo """
#prva slika
imageA1 = cv2.imread("Apple1.jpg")
imageB1 = cv2.imread("BackGr1.jpg")

#druga slika
imageA2 = cv2.imread("Apple2.jpg")
imageB2 = cv2.imread("BackGr2.jpg")

grayA1,grayA2,diff1,diff2,thresh1,thresh2 = prepareImage(imageA1,imageB1,imageA2,imageB2)
cx,cy,w,h,areaVSp,axesL,FruitFound,largest_contours = findFruit(thresh1, imageA1,True)
indexO = decideFruitType(areaVSp,axesL) # index glede na katerega se odločimo o tipu rezanja


# prikaz slik z detektiranimi konturami
cv2.imshow("Upragovljena slika razlik", thresh1)
cv2.imshow("Detektiran sadež",cv2.drawContours(FruitFound, largest_contours, -1, (0,255,0), 3))
cv2.waitKey(0)

#poiščemo preslikavo med slikama, ustvarimo korespondečne pare točk, ustvarimo sliko ujemajočih točk v trenutni mapi
NumbOfCent =getFruitPOints(grayA1,grayA2)

#najdemo x,y,z koordinate detektiranega sadeža
fruitCenter = findFruit3Dpose(NumbOfCent)
Rot = GetFruitOrientation(largest_contours)
print("Rotacija sadeža",Rot)

#pošlji koordinate glede na tip sadeža (jabolko ne potrbuje orientacije)
if indexO== 0:
    send_vals(fruitCenter[0],fruitCenter[1],fruitCenter[2],Rot[0],Rot[1],Rot[2],1,1,1,1,1,1)
elif indexO == 1:
    send_vals(fruitCenter[0],fruitCenter[1],fruitCenter[2],0,0,0,1,1,1,1,1,1)

