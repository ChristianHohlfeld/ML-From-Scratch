"""Generate a vector-accurate A4 fiducial target for camera calibration."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'hardware'/'board'/'board.svg'


def marker_svg(num,x,y,sz=24):
    im=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),num,600)
    # 4x4 payload + 1-cell black border = 6x6 grid
    rects=[]
    for r in range(6):
        for c in range(6):
            if im[50+r*100,50+c*100]<128:
                rects.append(f'<rect x="{x+c*sz/6:g}" y="{y+r*sz/6:g}" width="{sz/6:g}" height="{sz/6:g}"/>')
    return '<g fill="black">'+''.join(rects)+'</g>'


def board():
    # A4, 100% print size. Coordinates from physical print top-left in mm.
    # Four marker centres: (30,33),(180,33),(180,183),(30,183).
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
           '<rect width="210" height="297" fill="white"/>']
    for i,(x,y) in enumerate([(18,21),(168,21),(168,171),(18,171)]):
        parts.append(marker_svg(i,x,y))
        parts.append(f'<text x="{x+12}" y="{y+28}" fill="black" font-size="3.5" text-anchor="middle">ID {i}</text>')
    parts += [
      '<g stroke="#cccccc" fill="none" stroke-width="0.15">',
      '<rect x="30" y="33" width="150" height="150" stroke-dasharray="1 2"/>',
      '<circle cx="70" cy="118" r="3" stroke="#333333"/>',
      '<path d="M64 118 H76 M70 112 V124" stroke="#333333"/>',
      '</g>',
      '<text x="105" y="202" font-size="4" text-anchor="middle" fill="#222">HDR GEAR AI v0.7 | A4 100% | Shoulder axis: crosshair</text>',
      '<text x="105" y="210" font-size="3" text-anchor="middle" fill="#555">Measure printed marker centre spacing: exactly 150 mm horizontally and vertically</text>',
      '<text x="105" y="218" font-size="3" text-anchor="middle" fill="#555">Flat top-down camera. World origin = crosshair (70 mm, 118 mm on paper).</text>',
      '</svg>']
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text('\n'.join(parts),encoding='utf-8')
    return DEST

if __name__=='__main__': print(board())
