#!/bin/env python
# -*- coding: utf-8; -*-
#
# (c) 2016 FABtotum, http://www.fabtotum.com
#
# This file is part of FABUI.
#
# FABUI is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# FABUI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FABUI.  If not, see <http://www.gnu.org/licenses/>.

# Import standard python module
import argparse
import time
import gettext

# Import external modules
import numpy as np
import cv2,cv

# Import internal modules

# Set up message catalog access
tr = gettext.translation('print', 'locale', fallback=True)
_ = tr.ugettext

def vectorize_image(image_file, tgt_width, tgt_height, levels = 6, invert = False, crop = ''):
    
    img = cv2.imread(image_file)   
    
    if crop:
        crop = crop.split(',')
        x1 = int(crop[0])
        x2 = x1 + int(crop[2])
        y1 = int(crop[1])
        y2 = y1 + int(crop[3])
        print "{0} {1} {2} {3}".format(x1, x2, y1, y2)
        img = img[x1:x2, y1:y2]
        #cv2.imwrite('cropped.png', img) 
    
    # ?
    Z = img.reshape((-1,3))
    # Convert to np.float32
    Z = np.float32(Z)
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Define number of levels
    K = levels
    ret,label,center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # Convert to B&W
    img = cv2.cvtColor(res2, cv.CV_BGR2GRAY) 

    # Invert image if requested
    if invert:
        img = 255 - img

    # save a preview for internal use
    cv2.imwrite('img_preprocess.png', img) 

    print "COMPLETED PREPROCESSING. check img_preprocess.png for preview"

    h, w = res2.shape[:2]

    print ">> H: ", h
    print ">> W: ", w

def main():
    # SETTING EXPECTED ARGUMENTS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_file",       help=_("Image file."))
    parser.add_argument("-o", "--output",   help=_("Output gcode file."),    default='laser.gcode')
    parser.add_argument("-l", "--levels",   help=_("Laser power levels"),    default=6)
    parser.add_argument("-W", "--width",    help=_("Engraving width"),       default=0)
    parser.add_argument("-H", "--height",    help=_("Engraving height"),      default=0)
    parser.add_argument("-i", "--invert",   action='store_true', help=_("Engraving height"),   default=False)
    parser.add_argument("-C", "--crop",   help=_("Crop image. Use x,y,w,h'. Example: -c 0,0,100,100"),   default='')
    
    # GET ARGUMENTS
    args = parser.parse_args()

    # INIT VARs
    gcode_file      = args.output
    image_file      = args.image_file
    target_width    = int(args.width)
    target_height   = int(args.height)
    levels          = int(args.levels)
    invert          = bool(args.invert)
    crop            = args.crop
    
    vectorize_image(image_file, target_width, target_height, levels, invert, crop)

if __name__ == "__main__":
    main()
