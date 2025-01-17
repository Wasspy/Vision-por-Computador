# -*- coding: utf-8 -*-
'''
Texture flow direction estimation.
Sample shows how cv2.cornerEigenValsAndVecs function can be used 
to estimate image texture flow direction.
Usage:
    texture_flow.py [<image>]
'''

import numpy as np
import cv2

if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = 'Tablero1.jpg'

    img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    eigen = cv2.cornerEigenValsAndVecs(gray, 15, 3)
    eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
    flow = eigen[:,:,2]

    vis = img.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
    
    for x, y in points:
        
       x = int(x)
       y = int(y)
       vx, vy = np.int32(flow[y, x]*d)
       cv2.line(vis, (x-vx, y-vy), (x+vx, y+vy), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('input', img)
    cv2.imshow('flow', vis)
    cv2.waitKey()