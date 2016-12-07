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

class LaserEngraver:
    def __init__(self, filename):
        pass
        
    def draw_hline(self, x1, x2, y, color = 255):
        pass

class DebugOutput:
    def __init__(self, filename, width, height, color = 0):
        self.filename = filename
        self.dbg_img = np.ones((height, width, 1), np.uint8)*color
    
    def draw_rect(self, x1, y1, x2, y2, color = 255):
        self.dbg_img[y1:y2+1, x1:x2] = color
        
    def draw_hline(self, x1, x2, y, color = 255):
        self.dbg_img[y:y+1, x1:x2] = color
    
    def finish(self):
        print "Saving output to file '{0}'".format(self.filename)
        cv2.imwrite(self.filename, self.dbg_img)

def preprocess_raster_image(image_file, target_width, target_height, dot_size, levels = 6, invert = False, crop = ''):
    
    img = cv2.imread(image_file)   
    
    #### Resize: BEGIN ####
    
    h, w = img.shape[:2]
    
    # Minimal dots per mm
    min_dpm = int(1 / dot_size)
    
    # Default scaling factor for Width
    sx = 1
    # Default scaling factor for Height
    sy = 1
    
    if target_width:
        print "== X =="
        # Dots per mm based on given parameters
        dpm_x = target_width / float(w)
        print dpm_x
        
        # If DPM is less then minimal, scale the image so that pixels
        # are of proper size
        if dpm_x < min_dpm:
            max_w = float(target_width * min_dpm)
            sx = float(w) / max_w
            new_w = int(w / sx)
            print new_w
            
    if target_height:
        print "== Y =="
        # Dots per mm based on given parameters
        dpm_y = target_height / float(h)
        print dpm_y

        # If DPM is less then minimal, scale the image so that pixels
        # are of proper size        
        if dpm_y < min_dpm:
            max_h = float(target_height * min_dpm)
            sy = float(h) / max_h
            new_h = int(h / sy)
            print new_h
    
    scale = max(sx,sy)
    
    print scale
        
    w = int(w / scale)
    h = int(h / scale)
    
    if target_width == 0:
        target_width =  (w * float(target_height)) / h
        
    if target_height == 0:
        target_height =  (h * float(target_width)) / w 
    
    # Resize image if needed
    if scale != 1:
        print "Resize to {0}x{1}".format(w,h)
        img = cv2.resize(img,(w, h), interpolation = cv2.INTER_CUBIC)
    
    #### Resize: END ####
    
    if crop:
        crop = crop.split(',')
        x1 = int(crop[0])
        x2 = x1 + int(crop[2])
        y1 = int(crop[1])
        y2 = y1 + int(crop[3])
        print "Crop {0} {1} {2} {3}".format(x1, x2, y1, y2)
        img = img[y1:y2, x1:x2]
        cv2.imwrite('cropped.png', img) 
    
    # ?
    Z = img.reshape((-1,3))
    # Convert to np.float32
    Z = np.float32(Z)
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Define number of levels
    K = levels
    ret,label,center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    #print type(center), center.shape
    
    shades_of_gray = np.ndarray( (center.shape[0], 1) , dtype=int)
    i = 0
    for c in center:
        x = c[0] * 0.3 + c[1] * 0.59 + c[2] * 0.11
        #~ print "intensity @ level {0} = {1}".format(i, int(x))
        shades_of_gray[i] = int(x)
        i += 1
    
    # Center is an array of cluster representative colors
    # label is an array of cluster labels for every pixel (label=[0..K-1] )
    h, w = img.shape[:2]
    
    flat_labels = label.flatten()
    
    res = shades_of_gray[flat_labels]
    res2 = res.reshape( (h,w,1) )
    img = res2
    # Convert to B&W
    #img = cv2.cvtColor(res2, cv.CV_BGR2GRAY) 

    # Invert image if requested
    if invert:
        img = 255 - img

    # save a preview for internal use
    cv2.imwrite('img_preprocess.png', img)
    
    
    sorted_geays = shades_of_gray.copy()
    sorted_geays.sort(axis=0)
    #~ print  shades_of_gray
    #~ print  sorted_geays
    
    mapped = range(shades_of_gray.shape[0])
    rmapped = range(shades_of_gray.shape[0])
    
    for i in xrange(sorted_geays.shape[0]):
        value = sorted_geays[i]
        idx = np.nonzero(shades_of_gray == value)
        j = int(idx[0])
        mapped[i] = j
        rmapped[j] = i
    
    results = {
        'level' : range(levels),
        'width' : w,
        'height' : h,
        'target_width' : target_width,
        'target_height' : target_height
    }
    
    print results['level']
    
    for lvl in xrange(levels):
        
        value = int(sorted_geays[ lvl ])
        
        if invert:
            value = 255 - value
        
        results['level'][lvl] = {
            'lines' : [],
            'value' : value,
        }
        
        mask = np.zeros((shades_of_gray.shape), dtype=np.int)
        mask[ rmapped[lvl] ] = 255
        
        res = mask[flat_labels]
        res2 = res.reshape( (h,w,1) )
        # Convert to B&W
        #img = cv2.cvtColor(res2, cv.CV_BGR2GRAY) 

        # Invert image if requested
        if invert:
            res2 = 255 - res2

        # save a preview for internal use
        cv2.imwrite('img_preprocess_{0}.png'.format(lvl), res2) 
        
    for ii in xrange(len(mapped)):
        results['level'][ii]['lines'] = []
    
    work_width = int(target_width / dot_size)
    work_height = int(target_height / dot_size)
    results['work_width'] = work_width
    results['work_height'] = work_height
    
    
    for y in xrange(h):
        for ii in xrange(len(mapped)):
            results['level'][ii]['lines'].append([])
        
        old_lbl = -1
        new_lbl = -1
        start_x = 0
        
        for x in xrange(w):
            idx = y * w + x
            new_lbl = flat_labels[idx]
            if new_lbl != old_lbl:
                if x > start_x:
                    lbl = mapped[old_lbl]
                    
                    x1 = int( float(start_x * work_width) / w )
                    x2 = int( float(x * work_width) / w )
                    
                    results['level'][lbl]['lines'][y].append( (x1, x2) )
                old_lbl = new_lbl
                start_x = x
                
        if  x >= start_x:
            lbl = mapped[old_lbl]
            x1 = int( float(start_x * work_width) / w )
            x2 = int( float(x * work_width) / w )
            results['level'][lbl]['lines'][y].append( (x1, x2) )
    
    return results

def vectorize_raster_image(labels, shades_of_gray, width, height, target_width, target_height, dot_size, levels):
    levels = range(levels)
    
    # Create a blank
    #dpm = int(width/target_width)
    dbg_w = int(target_width / dot_size)
    dbg_h = int(target_height / dot_size)

    print width, height
    print dbg_w, dbg_h
    dbg_img = np.ones((dbg_h, dbg_w, 1), np.uint8)
    
    skip = 1
    skip_cnt = 0
    
    for y in xrange(dbg_h):
        img_y = int((float(height) * y) / float(dbg_h))
        
        if skip_cnt != skip:        
            skip_cnt += 1
            
            for x in xrange(dbg_w):
                img_x = int((float(width) * x) / float(dbg_w))
                idx = (img_y*width)+img_x
                if labels[idx] == 1:
                    dbg_img[y][x] = 255
        else:
            skip_cnt = 0
            
    cv2.imwrite('result.png', dbg_img)
    
    return levels

def draw_result(output, data):
    work_width = data['work_width']
    width = data['width']
    work_height = data['work_height']
    height = data['height']
    
    lvl = 0
    for level in data['level']:
        value = data['level'][lvl]['value']
    
        for y in xrange(work_height):
            img_y = int((float(height) * y) / float(work_height))
            for line in data['level'][lvl]['lines'][img_y]:
                y1 = y
                x1 = line[0]
                x2 = line[1]
                output.draw_hline( x1, x2, y1, color = value )
            
        lvl += 1

def main():
    # SETTING EXPECTED ARGUMENTS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_file",       help=_("Image file."))
    parser.add_argument("-o", "--output",   help=_("Output gcode file."),    default='laser.gcode')
    # Image preprocessing
    parser.add_argument("-l", "--levels",   help=_("Laser power levels"),    default=6)
    parser.add_argument("-i", "--invert",   action='store_true', help=_("Engraving height"),   default=False)
    parser.add_argument("-C", "--crop",   help=_("Crop image. Use x,y,w,h'. Example: -c 0,0,100,100"),   default='')
    # Vectorization
    parser.add_argument("-W", "--width",    help=_("Engraving width"),        default=0)
    parser.add_argument("-H", "--height",    help=_("Engraving height"),      default=0)
    #parser.add_argument("--offset-x",    help=_("Engraving line x offset (mm)"),      default=0)
    parser.add_argument("--offset-y",    help=_("Engraving line y offset (mm)"),   default=0)
    parser.add_argument("--line-width",  help=_("Engraving line width (mm)"),      default=0)
    #parser.add_argument("--line-height", help=_("Engraving line height (mm)"),    default=0)
    parser.add_argument("-D", "--dot-size",    help=_("Engraving dot size (mm)"),  default=0.1)
    parser.add_argument("-S", "--shortest-line",    help=_("Ignore lines shorter then this value (mm)"),  default=0.0)
    
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
    dot_size        = float(args.dot_size)
    shortest_line   = float(args.shortest_line)
    
    #labels, grays, w, h, tw, th = preprocess_raster_image(image_file, target_width, target_height, dot_size, levels, invert, crop)
    
    #~ preprocess_raster_image2(image_file, target_width, target_height, dot_size, levels, invert, crop)
    #print "w,h,tw,th", w,h,tw,th
    #~ img = cv2.imread('img_preprocess.png')  
    #~ h, w = img.shape[:2]
    
    #vectorize_raster_image(labels, grays, w, h, tw, th, dot_size, levels)
    result = preprocess_raster_image(image_file, target_width, target_height, dot_size, levels, invert, crop)
    dbg = DebugOutput('debug.png', result['work_width'], result['work_height'])
    draw_result(dbg, result)
    
    #~ dbg = DebugOutput('debug.png', 100, 100, 255)
    #~ dbg.draw_hline(0, 1, 50, color=0)
    #~ dbg.draw_hline(1, 2, 50, color=128)
    #~ dbg.draw_hline(2, 3, 50, color=0)
    dbg.finish()
    
if __name__ == "__main__":
    main()
