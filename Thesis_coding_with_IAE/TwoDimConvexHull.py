
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
from CommonDefs import Point


# In[20]:


class TwoDimConvexHull():
    def __init__(self, points):
        self.points = []
        self.points.extend(points)
    
    # Function takes the list of points and returns the index of the leftmost point
    def leftmostPointIndex(self):
        minn = 0
        for i in range(1,len(self.points)):
            if self.points[i].x < self.points[minn].x:
                minn = i
            elif self.points[i].x == self.points[minn].x:
                if self.points[i].y < self.points[minn].y:
                    minn = i
        return minn
    
    
    def orientation(self, p, q, r):
        '''
        To find orientation of ordered triplet (p, q, r).
        The function returns following values
        0 --> p, q and r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
        '''
        val = (q.y - p.y) * (r.x - q.x) -               (q.x - p.x) * (r.y - q.y)

        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2
    
    
    def getConvexHull(self):
     
        # There must be at least 3 points
        n = len(self.points)
        if n < 3:
            return

        # Find the leftmost point
        l = self.leftmostPointIndex()

        hull = []

        '''
        Start from leftmost point, keep moving counterclockwise
        until reach the start point again. This loop runs O(h)
        times where h is number of points in result or output.
        '''
        p = l
        q = 0
        while(True):

            # Add current point to result
            hull.append(self.points[p])

            '''
            Search for a point 'q' such that orientation(p, q,
            x) is counterclockwise for all points 'x'. The idea
            is to keep track of last visited most counterclock-
            wise point in q. If any point 'i' is more counterclock-
            wise than q, then update q.
            '''
            q = (p + 1) % n

            for i in range(n):

                # If i is more counterclockwise
                # than current q, then update q
                if(self.orientation(self.points[p],
                               self.points[i], self.points[q]) == 2):
                    q = i

            '''
            Now q is the most counterclockwise with respect to p
            Set p as q for next iteration, so that q is added to
            result 'hull'
            '''
            p = q

            # While we don't come to first point
            if(p == l):
                break

        # Print Result
    #    for pt in hull:
    #        print(pt.x, pt.y)
        return hull


# In[21]:


class PrintTwoDimConvexHull():
    def __init__(self, hull):
        xCoords = [pt.x for pt in hull]
        xCoords.append(hull[0].x)
        ycoords = [pt.y for pt in hull]
        ycoords.append(hull[0].y)
        ptIds = [pt.pid for pt in hull]
        ptIds.append(hull[0].pid)
        plt.plot(xCoords, ycoords, linewidth=2)
        for i, j, ptId in zip(xCoords, ycoords, ptIds):
            plt.text(i+0.01, j+0.02, '{}'.format(ptId))
        plt.show()

