import unittest
from pathlib import Path
import numpy as np
import cv2
from gear_ai.vision import detect_homography,detect_tip,MARKER_MM
from scripts.create_board import marker_svg,board

class Vision(unittest.TestCase):
 def make_frame(self,with_red=True,missing=()):
  # Synthetic perspective with fiducials, independently checked against
  # known A4 marker centres. THIS IS NOT A REAL CAMERA MEASUREMENT.
  canvas=np.full((1100,800,3),255,np.uint8)
  origins=[(30,33),(180,33),(180,183),(30,183)]
  for i,(mx,my) in enumerate(origins):
   if i in missing:continue
   marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),i,72)
   x,y=int((mx-12)*3),int((my-12)*3)
   canvas[y:y+72,x:x+72]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
  if with_red:
   cv2.circle(canvas,(int((70+40)*3),int((118-20)*3)),12,(0,0,255),-1)
  src=np.float32([[0,0],[800,0],[800,1100],[0,1100]])
  dst=np.float32([[28,17],[770,48],[745,1075],[55,1050]])
  return cv2.warpPerspective(canvas,cv2.getPerspectiveTransform(src,dst),(800,1100),borderValue=(255,255,255))
 def test_markers_homography_camera(self):
  frame=self.make_frame()
  H,e=detect_homography(frame)
  self.assertLess(e,.75)
  tip=detect_tip(frame,H)
  np.testing.assert_allclose(tip,[40,20],atol=2.5)
 def test_missing_marker_refused(self):
  with self.assertRaisesRegex(ValueError,'incomplete'):
   detect_homography(self.make_frame(missing=(3,)))
 def test_missing_target_refused(self):
  H,e=detect_homography(self.make_frame(with_red=False))
  with self.assertRaisesRegex(ValueError,'not detected'):
   detect_tip(self.make_frame(with_red=False),H)
 def test_vector_board_has_four_markers(self):
  svg=board().read_text()
  self.assertIn('width="210mm"',svg)
  self.assertEqual(svg.count('ID '),4)
 def test_source_labels(self):
  from scripts.digital_twin import run
  self.assertIn('SYNTHETIC',run()['source'])
if __name__=='__main__':unittest.main()
