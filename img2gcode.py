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
import json

# Import external modules
import numpy as np
import cv2,cv

# Import internal modules

# Set up message catalog access
tr = gettext.translation('img2gcode', 'locale', fallback=True)
_ = tr.ugettext

class EngraverOutput(object):
    
    def __init__(self, preset):
        self.preset = preset
        self.interleave = False
        
        self.skip_function = {
            'modulo' : self.get_y_list_modulo,
        }
    
    def draw_with_hlines(self, data):
        """
        Draw horizontal lines.
        """
        work_width = data['work_width']
        width = data['width']
        work_height = data['work_height']
        height = data['height']
        
        reverse = False
        
        lvl = 0
        for level in data['level']:
            value = data['level'][lvl]['value']
            #~ percent = data['level'][lvl]['percentage']
        
            self.comment(" Level {0}".format(lvl))
            self.level(value)
            
            old_img_y = -1
            y1 = 0
            
            if 'interleave' in self.preset['skip']:
                if self.preset['skip']['interleave']:
                    self.interleave = not self.interleave
            
            for y in xrange(work_height):
                # Get the pixel row corresponding to the burning y
                # Note: this is needed as a pixel might contain multiple laser lines (fat pixels)
                img_y = int((float(height) * y * 100.0) / float(work_height)) / 100
                
                # Detect the boundry of the pixel rows and create a draw_rect
                # command from it. This way it's easier to handle it as a bulk command.
                if old_img_y != img_y:
                    
                    self.comment(' Row {0}'.format(y))
                    
                    old_img_y = img_y
                    # Update row end
                    y2 = y+1
                    did_something = False
                
                    # Get all the lines in this row
                    lines = data['level'][lvl]['lines'][img_y]
                    
                    # Change the direction of burning to reduce laser movement
                    if reverse:
                        lines = reversed(lines)
                
                    for line in lines:
                        if reverse:
                            # Swap x1/x2 if reversed
                            x1 = line[1]
                            x2 = line[0]
                        else:
                            x1 = line[0]
                            x2 = line[1]
                        
                        #print x1, y1, x2, y2
                        self.draw_rect( x1, y1, x2, y2, value )
                        did_something = True
                    
                    # If any burning happened, reverse the burning direction
                    if did_something:
                        reverse = not reverse
                    
                    # Next row starts where this one ended
                    y1 = y2
                
            lvl += 1
            
    
    def get_y_list_modulo(self, color, y1, y2):
        """
        Generate a list of Y for drawing a primitive between y1 and y2.
        If lines have to skipped, only the Y values that have to be drawn
        will be in the output list.
        
        :param color:
        :param y1:
        :param y2:
        :type color: uint8
        :type y1: float
        :type y2: float
        :returns: List of Y values that should be drawn
        :rtype: list
        """
        in_list = range(y1,y2)
        out_list = []
        mod = self.preset['skip']['mod']
        on = self.preset['skip']['on']
        for i in in_list:
            if self.interleave:
                v = (i+1) % mod
            else:
                v = i % mod
            if v in on:
                out_list.append(i)
        
        return out_list
    
    def draw_rect(self, x1, y1, x2, y2, color = 0):
        """
        Draw a rectangle out of hlines. Skip lines if needed.
        """
        y_list = self.skip_function[ self.preset['skip']['type'] ](color, y1, y2)
        
        for y in y_list:
            self.draw_hline(x1, x2, y, color)
        
    def draw_hline(self, x1, x2, y, color = 0):
        """
        Draw a horizontal line.
        
        :param x1:    Start X
        :param x2:    End X
        :param y:     Start and end Y
        :param color: Line color
        """
        raise NotImplementedError('"draw_hline" function must be implemented')
        
    def start(self):
        """
        Engraver initialization callback.
        """
        pass
    
    def level(self, color):
        """
        Level change callback.
        
        :param color: Level color
        :type color: uint8
        """
        pass
    
    def comment(self, comment):
        """
        Line comment callback.
        """
        pass
        
    def end(self):
        """
        Engraver finalization callback.
        """
        pass

class LaserEngraver(EngraverOutput):
    """
    Laser engraver gcode generator class.
    """
    
    def __init__(self, filename, dot_size, preset):
        self.filename = filename
        self.dot_size = dot_size
        self.cur_x = 0.0
        self.cur_y = 0.0
        super(LaserEngraver, self).__init__(preset)
        
        self.pwm_function = {
            'const' : self.get_pwm_value_const,
            'linear' : self.get_pwm_value_linear,
        }
        
        self.speed_function = {
            'const' : self.get_burn_speed_value_const,
            'linear' : self.get_burn_speed_value_linear,
            'travel' : self.get_travel_speed_value_const
        }
        
        self.get_pwm_value = self.pwm_function[ preset['pwm']['type'] ]
        self.get_burn_speed = self.speed_function[ preset['speed']['type'] ]
        self.get_travel_speed = self.speed_function[ 'travel' ]

    def comment(self, comment):
        """
        Add comment to the gcode output.
        :param comment: Comment to be added.
        
        """
        self.fd.write(";{0}\r\n".format(comment))
    
    def get_pwm_value_const(self, color):
        """
        Returns constant PWM value.
        
        :returns: PWM value
        :rtype: uint8
        """
        return self.preset['pwm']['value']
        
    def get_pwm_value_linear(self, color):
        """
        Returns PWM value using a linear function to convert color
        to PWM value.
        
        :param color:
        :type color: uint8
        :returns: PWM value.
        :rtype: uint8
        """
        x_min = self.preset['pwm']['in-min']
        x_max = self.preset['pwm']['in-max']
        y_min = self.preset['pwm']['out-min']
        y_max = self.preset['pwm']['out-max']
        
        dx = float(x_max - x_min)
        dy = float(y_max - y_min)
        k = dy / dx

        y = 0

        if color >= x_min and color < x_max:
            y = y_min + (color-x_min) * k
            
        if color >= x_max:
            y = y_max

        return int(y)
        
    def get_burn_speed_value_const(self, color):
        """
        Returns constant burning speed.
        
        :returns: Burn speed value.
        :rtype: uint8
        """
        return self.preset['speed']['burn']
        
    def get_burn_speed_value_linear(self, color):
        """
        Returns burning speed using a linear function to convert color
        to speed value.
        
        :param color:
        :type color: uint8
        :returns: Burn speed value.
        :rtype: int
        """
        x_min = self.preset['speed']['in-min']
        x_max = self.preset['speed']['in-max']
        y_min = self.preset['speed']['out-min']
        y_max = self.preset['speed']['out-max']
        
        dx = float(x_max - x_min)
        dy = float(y_max - y_min)
        k = dy / dx

        y = 0

        if color >= x_min and color < x_max:
            y = y_min + (color-x_min) * k
            
        if color >= x_max:
            y = y_max

        return int(y)
                
    def get_travel_speed_value_const(self, color = 0):
        """
        Returns constant Travel speed.
        
        :returns: Travel speed.
        :rtype: uint8
        """
        return self.preset['speed']['travel']
    
    def level(self, color):
        """
        Level change setup gcode.
        """
        pwm = self.get_pwm_value(color)
        speed = self.get_burn_speed(color)
        print "Color {0} => PWM: {1}, Speed: {2}".format(color, pwm, speed)
        self.comment(' Applying PWM value {0}'.format(pwm) )
        self.fd.write('M3 S{0}\r\n'.format(pwm))
        self.fd.write('M400 ;Make sure all previous moves are finished\r\n')
    
    def start(self):
        """
        Engraver start function.
        """
        self.fd = open(self.filename, 'w')
        self.add_start_code()

    def add_start_code(self):
        """
        Engraver start code.
        """
        now = time.strftime("%c")
        self.fd.write("""\
;FABtotum laser engraving, coded on {0}
G4 S1 ;1 millisecond pause to buffer the bep bep
M450 S2 ; Activate laser module
M728 ;FAB bep bep
G90 ; absolute mode
G4 S1 ;1 second pause to reach the printer (run fast)
G1 F10000 ;Set travel speed
M107
""".format(now))
        
    def add_engrave_move(self, x, y, color):
        """
        Add gcode for an engraving move.
        
        :param x: Target X.
        :param y: Target y.
        :param color: Engrave color.
        :type x: float
        :type y: float
        :type color: uint8
        """
        feed = self.get_burn_speed(color)
        self.fd.write("G1 X{0} Y{1} F{2}\r\n".format(x, y, feed) )
        self.cur_x = x
        self.cur_y = y
        
    def add_travel_move(self, x, y):
        """
        Add gcode for a travel move.
        
        :param x: Target X
        :param y: Target y
        :type x: float
        :type y: float
        """
        feed = self.get_travel_speed()
        if x != self.cur_x or y != self.cur_y:
            self.fd.write("G0 X{0} Y{1} F{2}\r\n".format(x, y, feed) )
            self.cur_x = x
            self.cur_y = y
        
    def end(self):
        """
        Engraver end function.
        """
        self.add_end_code()
        self.fd.close()
        
    def add_end_code(self):
        """
        Shutdown code.
        """
        self.fd.write("""\
M400 ;Wait for all moves to finish
M728 ;FAB bep bep (end print)
G4 S1 ;pause
M5 ;shutdown
""")
    
    def draw_hline(self, x1, x2, y, color = 0):
        """
        Horizontal line callback.
        """
        real_x1 = x1 * self.dot_size
        real_x2 = x2 * self.dot_size
        real_y = y * self.dot_size
        
        self.add_travel_move(real_x1, real_y)
        self.add_engrave_move(real_x2, real_y, color)
        
class DebugOutput(EngraverOutput):
    """
    Engrever Debug Output class. Stores the result in an image with
    resolution of one dot per pixel.
    """
    
    def __init__(self, filename, width, height, color = 255, dot_size = 0.1, preset = {}):
        self.filename = filename
        self.dot_size = dot_size
        self.dbg_img = np.ones((height, width, 1), np.uint8)*color
        super(DebugOutput, self).__init__(preset)
        
    def draw_hline(self, x1, x2, y, color = 0):
        tx1 = min(x1,x2)
        tx2 = max(x1,x2)
        self.dbg_img[y:y+1, tx1:tx2] = color
    
    def end(self):
        print "Saving output to file '{0}'".format(self.filename)
        cv2.imwrite(self.filename, self.dbg_img)

def preprocess_raster_image(image_file, target_width, target_height, dot_size, levels = 6, invert = False, crop = ''):
    """
    Convert a raster image to horizontal line list classified into levels by color intensity.
    Only width or height has to be non-zero as the other value is automatically calculated
    based on the image width/height ration.
    
    :param image_file: Raster image filename.
    :param target_width: Target width in mm.
    :param target_height: Target height in mm.
    :param dot_size: Smallest engraver detail (mm).
    :param levels: Number of gray levels.
    :param invert: Invert gray intensity.
    :param crop: Crop image (x,y,w,h) (pixels)
    """
    
    img = cv2.imread(image_file)   
    
    #### Resize: BEGIN ####
    
    h, w = img.shape[:2]
    
    # Minimal dots per mm
    min_dpm = 1 #int(1 / dot_size)
    
    dpm = int(1 / dot_size)
    
    # Default scaling factor for Width
    sx = 1
    # Default scaling factor for Height
    sy = 1
    
    new_w = w
    new_h = h
    
    if target_width:
        print "== X =="
        # Dots per mm based on given parameters
        dpm_x = target_width / float(w)
        print "dpm_x", dpm_x * dpm
        print "dpm_x", round(dpm_x * dpm)
        
        # If DPM is less then minimal, scale the image so that pixels
        # are of proper size
        if dpm_x < 1.0:
            max_w = float(target_width)
            new_w = int( w * dpm_x * dpm )
            sx = 1.0 / (dpm_x * dpm)
            print "new_w", new_w
            
    if target_height:
        print "== Y =="
        # Dots per mm based on given parameters
        dpm_y = float(target_height) / float(h)
        ppm_y = float(h) / float(target_height)
        print "dpm", dpm
        print "dpm_y", dpm_y * dpm
        print "dpm_y", round(dpm_y * dpm)
        print "ppm_y", ppm_y

        # If DPM is less then minimal, scale the image so that pixels
        # are of proper size        
        if dpm_y < 1.0:
            max_h = float(target_height)
            new_h = int( h * dpm_y * dpm )
            sy = 1.0 / (dpm_y * dpm)
            print "new_h", new_h
    
    #scale = max(sx,sy)
    sx = w / float(new_w)
    sy = h / float(new_h)
    scale = max(sx,sy)
    print "scale", scale
        
    w = int(w / scale)
    h = int(h / scale)
    
    if target_width == 0:
        target_width =  (w * float(target_height)) / h
        
    if target_height == 0:
        target_height =  (h * float(target_width)) / w 
    
    if scale > 1:
        work_width = w
        work_height = h
    else:
        work_width = int(target_width / dot_size)
        work_height = int(target_height / dot_size)
    
    print "Target (mm)", target_width, target_height
    
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
    
    # Flip image on the Y axis to compensate for image Y axis and 
    # machine Y axis orientation
    img = cv2.flip(img,0)
    
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
        if invert:
            x = 255 - int(x)
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
    
    for lvl in xrange(levels):
        
        value = int(shades_of_gray[ rmapped[lvl] ])
        
        results['level'][lvl] = {
            'lines' : [],
            'value' : value,
            'percentage' : float(value) / 255.0,
        }
        
        mask = np.zeros((shades_of_gray.shape), dtype=np.int)
        mask[ lvl ] = 255
        
        res = mask[flat_labels]
        res2 = res.reshape( (h,w,1) )

        # Invert image if requested
        if invert:
            res2 = 255 - res2

        # save a preview for internal use
        cv2.imwrite('img_preprocess_{0}.png'.format(lvl), res2) 
        
    for ii in xrange(len(mapped)):
        results['level'][ii]['lines'] = []
    
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
            x2 = int( float(x * work_width) / w )+1
            results['level'][lbl]['lines'][y].append( (x1, x2) )
    
    return results

def main():
    # SETTING EXPECTED ARGUMENTS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_file",       help=_("Image file (jpg or png)."))
    parser.add_argument("preset_file",       help=_("Preset file (json)."))
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
    preset_file      = args.preset_file
    target_width    = int(args.width)
    target_height   = int(args.height)
    levels          = int(args.levels)
    invert          = bool(args.invert)
    crop            = args.crop
    dot_size        = float(args.dot_size)
    shortest_line   = float(args.shortest_line)
    
    result = preprocess_raster_image(image_file, target_width, target_height, dot_size, levels, invert, crop)
    
    #~ print "work (dots): {0}x{1}".format(result['work_width'], result['work_height'])
    
    with open(preset_file) as f:
        preset = json.load(f)
    
    lsr = LaserEngraver('output.gcode', dot_size, preset)
    lsr.start()
    lsr.draw_with_hlines(result)
    lsr.end()
    
    dbg = DebugOutput('debug.png', result['work_width'], result['work_height'], 0, dot_size, preset)
    dbg.start()
    dbg.draw_with_hlines(result)
    dbg.end()
    
if __name__ == "__main__":
    main()
