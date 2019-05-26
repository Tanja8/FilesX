import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float64)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('KalibracijaKamere/*.jpg')
#print(images)


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #print("KORNERSSSSSSSSSSS",corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
objpoints =objpoints.astype('float32')
imgpoints =imgpoints.astype('float32')

#print(objpoints,imgpoints)

camera = np.eye(3,3)
distortion = np.zeros(8)
rvecs = []
tvecs = []

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1],None,None)
ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],camera,distortion,rvecs,tvecs,0)

print("distorzijski koeficineti", mtx)


"shranim za kasejšo uporabo preračuna"
np.savetxt('CameraK.txt', mtx)