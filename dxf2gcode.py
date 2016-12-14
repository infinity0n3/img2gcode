# Import standard python module
import argparse
import time
import gettext
import json

# Import external modules
import numpy as np
import cv2,cv
import dxfgrabber

class Drawing(object):
    def __init__(self):
        self.primitives = []
        self.max_x = 0
        self.max_y = 0
        self.min_x = 0
        self.min_y = 0
    
    def extendBounds(self, points):
        
        if type(points) is list:
            pass
        elif type(points) is tuple:
            points = [ points ]
        else:
            print 'neither a tuple or a list'
            return
        
        for pt in points:
            if pt[0] > self.max_x:
                self.max_x = pt[0]
                
            if pt[0] < self.min_x:
                self.min_x = pt[0]
                
            if pt[1] > self.max_y:
                self.max_y = pt[1]
                
            if pt[1] < self.min_y:
                self.min_y = pt[1]
    
    def addLine(self, start, end):
        data = { 'type' : 'line', 'points': [start, end] }
        self.primitives.append(data)
        
        self.extendBounds(start)
        self.extendBounds(end)
        
    def addPolyline(self, points, closed = False):
        if closed:
            points.append(points[0])
        data = { 'type' : 'polyline', 'points' : points, 'closed' : closed }
        self.primitives.append(data)
        
        self.extendBounds(points)
        
    def addCircle(self, center, radius):
        data = { 'type' : 'circle', 'center' : center, 'radius' : radius }
        self.primitives.append(data)
        points = [
            (center[0] + radius, center[1] + radius),
            (center[0] - radius, center[1] - radius),
        ]
        self.extendBounds(points)
    
    def addArc(self, center, radius, start, end):
        data = { 'type' : 'arc', 'center' : center, 'radius' : radius, 'start' : start, 'end': end }
        self.primitives.append(data)
        # TODO: extendBounds
    
    def __rbasis(self, c, t, npts, x, h):
        """
        Generates rational B-spline basis functions for an open knot vector.
        :note: Source code converted from LibreCad (rs_spline.cpp)
        
        """
        nplusc = npts + c
        temp = np.zeros(nplusc)
        
        # calculate the first order nonrational basis functions n[i]
        for i in xrange(nplusc-1):
            if t >= x[i] and t < x[i+1]:
                temp[i] = 1

        # calculate the higher order nonrational basis functions
        for k in xrange(2,c+1):
            for i in xrange(nplusc-k):
                # if the lower order basis function is zero skip the calculation
                if temp[i] != 0:
                    temp[i] = ((t-x[i])*temp[i])/(x[i+k-1]-x[i])
                    
                # if the lower order basis function is zero skip the calculation
                if temp[i+1] != 0:
                    temp[i] += ((x[i+k]-t)*temp[i+1])/(x[i+k]-x[i+1])
                    
        # pick up last point
        if t >= x[nplusc-1]:
            temp[npts-1] = 1

        # calculate sum for denominator of rational basis functions
        sum = 0.0
        for i in xrange(npts):
            sum += temp[i]*h[i]

        r = np.zeros(npts)
        # form rational basis functions and put in r vector
        if sum != 0:
            for i in xrange(npts):
                r[i] = (temp[i]*h[i])/sum
        return r

    def __rbspline(self, npts, k, p1, b, knot):
        """
        Generates a rational B-spline curve using a uniform open knot vector.
        :note: Source code converted from LibreCad (rs_spline.cpp)
        
        :param npts: Number of control points
        :param k: Spline degree
        :param b: Control point list
        :param knot: knot list
        """
        p = []
        h = np.ones(npts+1)
        nplusc = npts + k

        # generate the open knot vector (we have one already)
        x = knot

        # calculate the points on the rational B-spline curve
        t = 0.0
        step = x[nplusc-1] / (p1-1)
            
        vp = np.zeros(shape=(p1,2))
            
        for i in xrange(p1):
            if x[nplusc-1] - t < 5e-6:
                t = x[nplusc-1]
            # generate the basis function for this value of t
            nbasis = self.__rbasis(k, t, npts, x, h)

            # generate a point on the curve
            for j in xrange(npts):
                x0 = b[j][0] * nbasis[j]
                y0 = b[j][1] * nbasis[j]
                vp[i] += ( x0, y0 )
                
            t += step
            
            p.append( vp[i] )
            
        return p
    
    def addSpline(self, control_points, knots, degree):
        
        npts = len(control_points)
        k = degree + 1
        #~ p1 = (dxf.header['$SPLINESEGS'] or 8) * npts
        p1 = (8) * npts
        
        points = self.__rbspline(npts, k, p1, control_points, knots)
        
        data = { 'type' : 'spline', 'control_points' : control_points, 'knots' : knots, 'degree' : degree, 'points' : points}
        self.primitives.append(data)
        # TODO: extendBounds
    
    def transform(self, sx = 1.0, sy = 1.0, ox = 0.0, oy = 0.0):
        self.max_x *= sx
        self.max_y *= sy
        self.min_x *= sx
        self.min_y *= sy
        
        for e in self.primitives:
            t = e['type']
            if t == 'polyline' or t == 'spline':
                points = []
                for p in e['points']:
                    points.append( ( (ox + p[0])*sx, (oy + p[1])*sy) )
                e['points'] = points
            elif t == 'circle' or t == 'arc':
                p = e['center']
                e['center'] = ( (ox + p[0])*sx, (oy + p[1])*sy)
                e['radius'] *= (sx+sy) / 2.0
        
    def scale(self, sx = 1.0, sy = 1.0):
        self.transform(sx, sy)
        
def preprocess_dxf_image(filename):
    dxf = dxfgrabber.readfile(filename)
    print(dxf.header['$SPLINESEGS'])

    output = Drawing()

    print "- Entities:"
    for e in dxf.entities:
        t = e.dxftype
        print "== ", t
        if t == 'LWPOLYLINE' or t == 'POLYLINE':
            is_closed = False
            if e.is_closed and t == 'POLYLINE':
                is_closed = True
            
            output.addPolyline(e.points, is_closed)
            
        elif t == 'LINE':
            output.addLine(e.start, e.end)
            
        elif t == 'CIRCLE':
            output.addCircle(e.center, e.radius)
            
        elif t == 'ARC':
            output.addArc(e.center, e.radius, e.start_angle, e.end_angle)
            
        elif t == 'SPLINE':
            
            output.addSpline(e.control_points, e.knots, e.degree)
            
            #~ print e.degree
            #~ print e.start_tangent
            #~ print e.end_tangent
            #~ print e.control_points
            #~ print e.fit_points
            #~ print e.knots
            #~ print e.weights
            #~ print e.normal_vector
    
    return output

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
                        self.fill_rect( x1, y1, x2, y2, value )
                        did_something = True
                    
                    # If any burning happened, reverse the burning direction
                    if did_something:
                        reverse = not reverse
                    
                    # Next row starts where this one ended
                    y1 = y2
                
            lvl += 1
            
    def draw(self, data):
        print 'draw-area', data.min_x, data.min_y, data.max_x, data.max_y
        width = data.max_x - data.min_x
        height = data.max_y - data.min_y
        print 'draw', width, height
        print 'offset', data.min_x, data.min_y
        
        for e in data.primitives:
            t = e['type']
            if t == 'line':
                pass
            elif t == 'polyline' or t == 'spline':
                self.draw_polyline(e['points'])
            elif t == 'circle':
                c = e['center']
                self.draw_circle( c[0], c[1], e['radius'])
            elif t == 'arc':
                c = e['center']
                self.draw_arc( c[0], c[1], e['radius'], e['start'], e['end'])
            #~ elif t == 'spline':
                #~ self.draw_spline(e['control_points'], e['knots'], e['degree'])
            else:
                print 'TODO', t
    
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
    
    def fill_rect(self, x1, y1, x2, y2, color = 0):
        """
        Draw a rectangle out of hlines. Skip lines if needed.
        """
        y_list = self.skip_function[ self.preset['skip']['type'] ](color, y1, y2)
        
        for y in y_list:
            self.draw_hline(x1, x2, y, color)
    
    def draw_polyline(self, points, color = 0):
        p0 = points[0]
        for p in points[1:]:
            p1 = ( int(p[0]), int(p[1]))
            #cv2.circle(self.dbg_img, p1, 2, 0)
            self.draw_line( p0[0], p0[1], p[0], p[1], color)
            p0 = p
    
    def draw_line(self, x1, y1, x2, y2, color = 0):
        raise NotImplementedError('"draw_line" function must be implemented')
    
    def draw_spline(self, control_points, knots, degree, color = 0):
        raise NotImplementedError('"draw_spline" function must be implemented')
    
    def draw_circle(self, x1, y1, r, color = 0):
        raise NotImplementedError('"draw_circle" function must be implemented')
        
    def draw_arc(self, x1, y1, r, start, end, color = 0):
        raise NotImplementedError('"draw_arc" function must be implemented')
    
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
    
    def draw_line(self, x1, y1, x2, y2, color = 0):
        cv2.line(self.dbg_img, (int(x1),int(y1)), (int(x2),int(y2)), color)
    
    def draw_circle(self, x1, y1, r, color = 0):
        cv2.circle(self.dbg_img, (int(x1), int(y1)), int(r), color)
    
    def draw_arc(self, x0, y0, r, start, end, color = 0, step = 10.0):
        """
        Draw an arc.
        """
        
        angle = end - start
        if angle < 0:
            angle += 360
        
        steps = int(abs(angle / step))
        
        x1 = x0 + np.cos( np.deg2rad(start) )*r
        y1 = y0 + np.sin( np.deg2rad(start) )*r
        
        for a in xrange(steps):
            angle = np.deg2rad(start + a*step)
            x2 = x0 + np.cos(angle)*r
            y2 = y0 + np.sin(angle)*r
            
            self.draw_line(x1,y1, x2,y2, color)

            x1 = x2
            y1 = y2
            
        if (start + (steps-1)*step) != end:
            angle = np.deg2rad(end)
            x2 = x0 + np.cos(angle)*r
            y2 = y0 + np.sin(angle)*r
            
            self.draw_line(x1,y1, x2,y2, color)
    
    def __point_on_line(self, p1, p2, t):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        x = p1[0] + float(dx*t)
        y = p1[1] + float(dy*t)
        
        #~ print 'p1',p1, 'p2',p2, 'dx', dx,  'dy',dy, 't',t
        
        return (x, y)
    
    def __sub_bezier(self, points, t):
        sub_points = []
        
        p1 = points[0]
        for p2 in points[1:]:
            pt = self.__point_on_line(p1, p2, t)
            sub_points.append(pt)
            p1 = p2
        
        if len(sub_points) > 1:
            return self.__sub_bezier(sub_points, t)
        else:
            return sub_points
    
    def __bezier_points(self, points, t):
        if type(t) == list or type(t) == np.ndarray:
            pts = []
            for t1 in t:
                p = self.__sub_bezier(points, t1)
                pts = pts + p
            return pts
        else:
            return self.__sub_bezier(points, t)
    
    def end(self):
        print "Saving output to file '{0}'".format(self.filename)
        cv2.imwrite(self.filename, self.dbg_img)
        
    def show(self):
        self.dbg_img = cv2.flip(self.dbg_img,0)
        cv2.imshow('image', self.dbg_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#~ drawing = preprocess_dxf_image("dxf_default.dxf")
drawing = preprocess_dxf_image("dxf_sample_laser.dxf")
#~ drawing = preprocess_dxf_image("librecad3.dxf")

drawing.transform(20, 20, -drawing.min_x+1, -drawing.min_y+1)

dbg = DebugOutput('draw.png', 500, 500)
dbg.start()
dbg.draw(drawing)


#~ dbg.end()
dbg.show()
