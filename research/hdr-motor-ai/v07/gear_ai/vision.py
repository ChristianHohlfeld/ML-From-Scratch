"""Optical measurement of a physical V0 tip, not kinematic reconstruction.

Uses four fixed ArUco fiducials on an A4 print sheet and a bright-red dot
at the physical linkage tip. Image processing is performed OFF-device by a
camera host. Ground truth positions come only from image evidence.
"""
from __future__ import annotations
import numpy as np

# Marker centre coordinates on the calibration A4 sheet, mm, origin at
# printed motor axis reference (70, 118), world y positive upward.
MARKER_MM={0:(-40.,85.),1:(110.,85.),2:(110.,-65.),3:(-40.,-65.)}


def _cv():
    try: import cv2
    except ImportError as exc: raise RuntimeError('pip install -e ".[vision]"') from exc
    if not hasattr(cv2,'aruco'): raise RuntimeError('OpenCV aruco module required')
    return cv2


def detect_homography(frame):
    """Return (H, reprojection_err_mm). Require all four unique fiducials."""
    cv=_cv()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    dictionary=cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    corners,ids,_=cv.aruco.ArucoDetector(dictionary,cv.aruco.DetectorParameters()).detectMarkers(gray)
    if ids is None: raise ValueError('No board markers detected')
    pts={}
    for poly,ind in zip(corners,ids.ravel()):
        if int(ind) in MARKER_MM:
            if int(ind) in pts:raise ValueError('Ambiguous duplicate calibration marker')
            pts[int(ind)]=np.mean(poly[0],axis=0)
    if set(pts)!=set(MARKER_MM):raise ValueError(f'Board incomplete: {sorted(pts)}')
    pix=np.array([pts[k] for k in range(4)],dtype=np.float32)
    mm=np.array([MARKER_MM[k] for k in range(4)],dtype=np.float32)
    if cv.contourArea(pix.reshape(-1,1,2))<4000:raise ValueError('Calibration markers too small')
    H=cv.getPerspectiveTransform(pix,mm)
    mapped=cv.perspectiveTransform(pix[None,:,:],H)[0]
    err=float(np.linalg.norm(mapped-mm,axis=1).max())
    if not (np.isfinite(H).all() and np.linalg.cond(H)<1e10 and err<0.25):
        raise ValueError('Unstable calibration')
    return H,err


def detect_tip(frame,H,min_area=18):
    """Optical red-dot centre in world mm. Does not use servo targets."""
    cv=_cv()
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask=cv.bitwise_or(cv.inRange(hsv,(0,90,70),(12,255,255)),
                       cv.inRange(hsv,(168,90,70),(179,255,255)))
    mask=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
    components,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cands=[c for c in components if cv.contourArea(c)>=min_area]
    if not cands:raise ValueError('Red optical target not detected')
    c=max(cands,key=cv.contourArea)
    area=cv.contourArea(c)
    if len(cands)>1 and cv.contourArea(sorted(cands,key=cv.contourArea)[-2])>area*.55:
        raise ValueError('Ambiguous multiple red targets')
    moments=cv.moments(c)
    if moments['m00']<1e-9:raise ValueError('Degenerate optical target')
    xy=np.array([[[moments['m10']/moments['m00'],moments['m01']/moments['m00']]]],dtype=np.float32)
    point=cv.perspectiveTransform(xy,H)[0,0]
    if not np.isfinite(point).all():raise ValueError('Non-finite optical coordinate')
    return point
