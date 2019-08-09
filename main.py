import cv2 as cv
import numpy as np
import math
import os
from obj import *

def render(image, obj, projection, model, color = None):
    vertices = obj.vertices
    scaleMatrix = np.eye(3)*3
    h, w = model.shape

    for face in obj.faces:
        faceVertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in faceVertices])
        points = np.dot(points, scaleMatrix)
        points = np.array([[i[0] + w/2, i[1] + h/2, i[2]] for i in points])
        transform = cv.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imagePoints = np.int32(transform)
        if color == None:
            cv.fillConvexPoly(image, imagePoints, (137, 27, 211))
        else:
            color = colorRgb(face[-1])
            color = color[::-1]
            cv.fillConvexPoly(image, imagePoints, color)

    return image

def projectionMatrix(cam, hom):
    hom = -hom
    rotTransform = np.dot(np.linalg.inv(cam), hom)
    col0 = rotTransform[:, 0]
    col1 = rotTransform[:, 1]
    col2 = rotTransform[:, 2]
    normal = math.sqrt(np.linalg.norm(col0, 2)*np.linalg.norm(col1, 2))
    rot0 = col0/normal
    rot1 = col1/normal
    translation = col2/normal
    c = rot0 + rot1
    p = np.cross(rot0, rot1)
    d = np.cross(c, p)
    rot0 = np.dot(c/np.linalg.norm(c, 2) + d/np.linalg.norm(d, 2), 1/math.sqrt(2))
    rot1 = np.dot(c/np.linalg.norm(c, 2) - d/np.linalg.norm(d, 2), 1/math.sqrt(2))
    rot2 = np.cross(rot0, rot1)
    projection = np.stack((rot0, rot1, rot2, translation)).T
    return np.dot(cam, projection)

def colorRgb(color):
    color = color.lstrip("#")
    colorLength = len(color)
    return ((int(color[i:i + colorLength//3], 16) for i in range(0, colorLength, colorLength//3)))

class Mesh:
    def __init__(self, obj):
        obj = OBJ(os.path.join(dirName, obj), swapyz = True)
        self.cap = cv.VideoCapture(0)

    minMatches = 10
    hom = None
    cam = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    dirName = os.getcwd()
    kpModel, desModel = orb.detectAndCompute(model, None)

    def parent(self, modelFile):
        model = cv.imread(os.path.join(dirName, modelFile), 0)

    def update(self):
        ret, frame = cap.read()
        kpFrame, desFrame = orb.detectAndCompute(frame, None)
        matches = bf.match(desModel, desFrame)
        matches = sorted(matches, key = lambda x: x.distance)

        if len(matches) > minMatches:
            sourcePoints = np.float32([kpModel[i.queryIdx].pt for i in matches]).reshape(-1, 1, 2)
            transformPoints = np.float32([kpFrame[i.trainIdx].pt for i in matches]).reshape(-1, 1, 2)
            hom, mask = cv.findHomography(sourcePoints, transformPoints, cv.RANSAC, 5.0)

            if args.rectangle:
                h, w = model.shape
                points = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transform = cv.perspectiveTransform(points, hom)
                frame = cv.polylines(frame, [np.int32(transform)], True, 255, 3, cv.LINE_AA)

            if hom is not None:
                try:
                    projection = projectionMatrix(cam, hom)
                    frame = render(frame, obj, projection, model, False)
                except:
                    pass

            if args.matches:
                frame = cv.drawMatches(model, kpModel, frame, kpFrame, matches[:10], 0, flags = 2)

            cv.imshow("frame", frame)
            key = cv.waitKey(30) & 0xff
            if key == 27:
                self.cap.release()
                cv.destroyAllWindows()
        else:
            cap.release()
            cv.destroyAllWindows()

class ReverseGreenscreen():
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.count = 0
        for i in range(100):
            value, self.background = self.cap.read()
            if value == False:
                continue

        self.background = np.flip(self.background, axis = 1)

    def screen(self, colors):
        self.colors = colors
        if len(self.colors) == 1:
            self.colors = [self.colors[0], self.colors[0], self.colors[0], self.colors[0]]
        elif len(self.colors) == 2:
            self.colors = [self.colors[0], self.colors[1], self.colors[0], self.colors[1]]

    def update(self):
        if cap.isOpened():
            value, self.image = cap.read()
            if not value:
                break
            self.count += 1
            self.image = np.flip(self.image, axis = 1)
            hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
            mask0 = cv.inRange(hsv, np.array(self.colors[0]), np.array(self.colors[1]))
            mask1 = cv.inRange(hsv, np.array(self.colors[2]), np.array(self.colors[3]))
            mask0 = mask0 + mask1
            mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 2)
            mask0 = cv.dilate(mask0, np.ones((3, 3), np.uint8), iterations = 1)
            mask1 = cv.bitwise_not(mask0)
            res0 = cv.bitwise_and(self.background, self.background, mask = mask0)
            res1 = cv.bitwise_and(self.image, self.image, mask = mask1)
            frame = cv.addWeighted(res0, 1, res1, 1, 0)

            cv.imshow("frame", frame)
            key = cv.waitKey(30) & 0xff
            if key == 27:
                self.cap.release()
                cv.destroyAllWindows()

class Greenscreen:
    def __init__(self, background):
        self.cap = cv.VideoCapture(0)
        self.background = background

    def screen(self, colors):
        self.colors = colors
        if len(self.colors) == 1:
            self.colors = [self.colors[0], self.colors[0], self.colors[0], self.colors[0]]
        elif len(self.colors) == 2:
            self.colors = [self.colors[0], self.colors[1], self.colors[0], self.colors[1]]

    def update(self):
        value, self.image = self.cap.read()
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        mask = cv.inRange(self.image, np.array([self.colors[0]]), np.array([self.colors[1]]))
        imageMask = np.copy(self.image)
        imageMask[mask != 0] = [0, 0, 0]
        background = cv.imread(self.background)
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        backgroundCrop = background[0:self.image[0], 0:self.image[1]]
        backgroundCrop[mask = 0] = [0, 0, 0]
        frame = backgroundCrop + imageMask

        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            cv.destroyAllWindows()

class RegionalEdit:
    def __init__(self, startPos, endPos):
        self.cap = cv.VideoCapture(0)
        self.startPos = {startPos[0][0]:startPos[0][1], startPos[1][0]:startPos[1][1]}
        self.endPos = {endPos[0][0]:endPos[0][1], endPos[1][0]:endPos[1][1]}

     def update(self):
        value, self.image = self.cap.read()
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        start = self.image[startPos]
        self.image[endPos] = start
        frame = self.image
        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            cv.destroyAllWindows()

class Filter:
    def __init__(self, filterName):
        self.filterName = filterName

    def update(self):
        value, self.image = self.cap.read()
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        if filterName == "grayscale":
            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)

class Hightlight:
     def __init__(self, background):
        self.cap = cv.VideoCapture(0)
        self.background = background

    def screen(self, colors):
        self.colors = colors
        if len(self.colors) == 1:
            self.colors = [self.colors[0], self.colors[0], self.colors[0], self.colors[0]]
        elif len(self.colors) == 2:
            self.colors = [self.colors[0], self.colors[1], self.colors[0], self.colors[1]]

    def update(self):
        value, self.image = self.cap.read()
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(self.image, np.array([self.colors[0]]), np.array([self.colors[1]]))
        frame = cv.bitwise_and(self.image, self.image, mask = mask)

        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            cv.destroyAllWindows()

class Detector:
     def __init__(self, cascadeFile):
        self.cap = cv.VideoCapture(0)
        self.cascade = cv.CascadeClassifier(shapeFile)

    def update(self):
        value, self.image = self.cap.read()
        grayscaleImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        cascades = self.cascade.detectMultiScale(grayscaleImage, 1.3, 5)
        for x, y, w, h in cascades:
            cv.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv.imshow("frame", self.image)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            cv.destroyAllWindows()

class Replace:
    def __init__(self, detectorObj, image):
        self.cap = cv.VideoCapture(0)
        self.detectorObj = cv.detectorObj
        self.image = image

    def update(self):
        value, image = self.cap.read()
        grayscaleImage = cv.cvtColor(self.detectorObj.image, cv.COLOR_BGR2GRAY)
        cascades = self.detectorObj.cascade.detectMultiScale(grayscaleImage, 1.3, 5)
        images = []
        for x, y, w, h in cascades:
            images.append({x:y, x + w:y + h})

        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

        for i in images:
            replacement = self.image[i]
            image[i] = replacement

        cv.imshow("frame", image)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            cv.destroyAllWindows()

class Clipping:
    def __init__(self, name, *videos):
        self.videos = videos
        output = cv.VideoWriter(name, cv.cv.CV_FOURCC("F", "M", "P", "4"), 15, (), 1)
        for video in range(len(videos))
            self.cap = cv.VideoCapture(self.videos[i])
            value, frame = self.cap.read()
            cv.imshow("frame", frame)
            output.write(self.cap)
            key = cv.waitKey(30) & 0xff
            if key == 27:
                self.cap.release()
                output.release()
                cv.destroyAllWindows()

class Player:
    def __init__(self, file):
        self.file = file
        self.cap = cv.VideoCapture(self.videos[i])
        while cap.isOpened():
            value, frame = self.cap.read()
            cv.imshow("frame", frame)
            key = cv.waitKey(30) & 0xff
            if key == 27:
                self.cap.release()
                cv.destroyAllWindows()

class Camera:
    def __init__(self, rot = (0, 0)):
        global camera
        self.x, self.y = rot
        camera = self

def Point(fov, x, y, z):
    global camera
    fov = 100
    near = fov/((((math.cos(camera.x)*y) + (math.sin(camera.x)*x)*math.sin(camera.y)) + fov) - (z*math.cos(camera.y)))
    x = ((math.cos(camera.x)*x) - (math.sin(camera.x)*y))*near
    y = (math.cos(camera.y)*((math.cos(camera.x)*y) + (math.sin(camera.x)*x)) + (z*math.sin(camera.y)))*near
    return x, y

class StillMesh:
    def __init__(self, file, fov = 100):
        self.file = file
        self.cap = cv.VideoCapture(self.videos[i])
        self.points = [OBJ(file).vertices]
        self.faces = [OBJ(file).faces]
        self.pos = pos
        self.fov = fov
        x, y, z = self.pos
        self.points = [(X + x, Y + y, Z + z) for X, Y, Z in self.points]

    def update(self):
        value, frame = self.cap.read()
        faces = []
        for face in self.faces:
            faceCurrent = []
            for i in face:
                faceCurrent.append((self.points[i]))
            faces.append((faceCurrent))
        for face in faces:
            points = []
            for i in face:
                points += list(Point(self.fov, *i))
            cv.polylines(frame, [np.array([points], np.int32).reshape((-1, 1, 2))], True, (255, 255, 255))
        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

    def scale(self, x = 1, y = 1, z = 1):
        self.points = [(X*x, Y*y, Z*z) for X, Y, Z in self.points]

    def move(self, x = 0, y = 0, z = 0):
        self.points = [(X + x, Y + y, Z + z) for X, Y, Z in self.points]

    def rotate(self, X = 0, Y = 0, Z = 0):
        points = []
        for x, y, z in self.points:
            rad = X*math.pi/180
            cosa = math.cos(rad)
            sina = math.sin(rad)
            y = y*cosa - z*sina
            z = y*sina + z*cosa
            rad = Y*math.pi/180
            cosa = math.cos(rad)
            sina = math.sin(rad)
            z = z*cosa - x*sina
            x = z*sina + x*cosa
            rad = Z*math.pi/180
            cosa = math.cos(rad)
            sina = math.sin(rad)
            x = x*cosa - y*sina
            y = x*sina + y*cosa
            points.append((x, y, z))
        self.points = points

    def displace(self, face = 0, amount = 1, damping = 1):
        face = self.faces[face]
        for i in face:
            self.points[i] = (self.points[i][0] + float(random.randint(-amount, amount))/damping, self.points[i][1] + float(random.randint(-amount, amount))/damping, self.points[i][2] + float(random.randint(-amount, amount))/damping)

    def subdivide(self, face = 0):
        pointsLength = len(self.points)
        face = self.faces[face]
        p1 = self.points[face[0]]
        p2 = self.points[face[1]]
        p3 = self.points[face[2]]
        p4 = self.points[face[3]]
        center = (((p1[0] * 0.25) + (p2[0] * 0.25) + (p3[0] * 0.25)), ((p1[1] * 0.25) + (p2[1] * 0.25) + (p3[1] * 0.25)), ((p1[2] * 0.25) + (p2[2] * 0.25) + (p3[2] * 0.25)))
        p12 = (((p1[0] * 0.5) + (p2[0] * 0.5)), ((p1[1] * 0.5) + (p2[1] * 0.5)), ((p1[2] * 0.5) + (p2[2] * 0.5)))
        p23 = (((p2[0] * 0.5) + (p3[0] * 0.5)), ((p2[1] * 0.5) + (p3[1] * 0.5)), ((p2[2] * 0.5) + (p3[2] * 0.5)))
        p34 = (((p3[0] * 0.5) + (p4[0] * 0.5)), ((p3[1] * 0.5) + (p4[1] * 0.5)), ((p3[2] * 0.5) + (p4[2] * 0.5)))
        p41 = (((p4[0] * 0.5) + (p1[0] * 0.5)), ((p4[1] * 0.5) + (p1[1] * 0.5)), ((p4[2] * 0.5) + (p1[2] * 0.5)))
        self.points += center, p12, p23, p34, p41
        f1 = (self.points.index(p1[-pointsLength:]), self.points.index(p12[-pointsLength:]), self.points.index(center[-pointsLength:]), self.points.index(p41[-pointsLength:]))
        f2 = (self.points.index(p12[-pointsLength:]), self.points.index(p2[-pointsLength:]), self.points.index(p23[-pointsLength:]), self.points.index(center[-pointsLength:]))
        f3 = (self.points.index(center[-pointsLength:]), self.points.index(p23[-pointsLength:]), self.points.index(p3[-pointsLength:]), self.points.index(p34[-pointsLength:]))
        f4 = (self.points.index(p41[-pointsLength:]), self.points.index(center[-pointsLength:]), self.points.index(p34[-pointsLength:]), self.points.index(p4[-pointsLength:]))
        self.faces.remove(face)
        self.faces += f1, f2, f3, f4

    def checkCollisions(self, obj):
        xMax = max([i[0] for i in self.points])
        xMin = min([i[0] for i in self.points])
        yMax = max([i[1] for i in self.points])
        yMin = min([i[1] for i in self.points])
        zMax = max([i[2] for i in self.points])
        zMin = min([i[2] for i in self.points])
        xMax2 = max([i[0] for i in obj.points])
        xMin2 = min([i[0] for i in obj.points])
        yMax2 = max([i[1] for i in obj.points])
        yMin2 = min([i[1] for i in obj.points])
        zMax2 = max([i[2] for i in obj.points])
        zMin2 = min([i[2] for i in obj.points])
        collision = ((xMax >= xMin2 and xMax <= xMax2) or (xMin <= xMax2 and xMin >= xMin2)) and ((yMax >= yMin2 and yMax <= yMax2) or (yMin <= yMax2 and yMin >= yMin2)) and ((zMax >= zMin2 and zMax <= zMax2) or (zMin <= zMax2 and zMin >= zMin2))
        return collision

    def pixelate(self, amount = 0):
        for i in range(amount):
            self.subdivide(random.randint(0, len(a.faces) - 1))

    def movePoint(self, point = 0, x = 0, y = 0, z = 0):
        self.points[point] = (self.points[point][0] + x, self.points[point][1] + y, self.points[point][2] + z)

def distance(p0, p1):
    return math.sqrt(((p1[0] - p0[0])**2) + ((p1[1] - p0[1])**2))

class Effect:
    #### lighting and other
        ### environment
            ### pixel manipulation
    ### color filtering
    ## image filtering
    ## painting
        ### after-effects
            ### pixel manipulation and shapes
    # blending images
    # smoothing and blurring
    pass

class Light(Effect):
    def __init__(self, pos, strength = 10):
        self.pos = pos
        self.strength = strength

    def update(self):
        value, frame = self.cap.read()
        height, width = frame.shape
        for h in range(height):
            for w in range(width):
                frame[h, w] = [frame[h, w][0] + (distance(frame[h, w][0], pos)*self.strength/10), frame[h, w][1], frame[h, w][2]]

        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

class Pixel(Effect):
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

    def update(self):
        value, frame = self.cap.read()
        frame[self.pos[0], self.pos[1]] = [color[0], color[1]]

        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

class Polygon(Effect):
    def __init__(self, points):
        self.points = points

    def update(self):
        value, frame = self.cap.read()
        cv.polylines(frame, [np.array([points], np.int32).reshape((-1, 1, 2))], True, (255, 255, 255))
        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

class Overlay:
    def __init__(self, pos, image):
        self.pos = pos
        self.image = cv.imread(image, -1)

    def update(self):
        value, frame = self.cap.read()
        y0, y1 = self.pos[0], self.pos[1] + self.image.shape[0]
        x0, x1 = self.pos[0], self.pos[1] + self.image.shape[1]
        alphaImage = self.image[:, :, 3]/255.0
        alphaFrame = 1.0 - alphaImage
        for i in range(0, 3):
            frame[y0:y1, x0:x1, i] = (alphaImage*self.image[:, :, c] + alphaFrame*frame[y0:y1, x0:x1, i])

        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

class WarpToFit:
    def __init__(self, points):
        self.points = np.float32([points])

    def update(self):
        value, frame = self.cap.read()
        points0 = self.points
        points1 = np.float32([[0, 0], [frame.shape[0], 0], [0, frame.shape[1]], [frame.shape[0], frame.shape[1]]])
        transform = cv.getPerspectiveTransform(points0, points1)
        frame = cv.warpPerspective(frame, transform, (frame.shape[0], frame.shape[1]))
        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()

class Warp:
    def __init__(self, points0, points1):
        self.points0 = np.float32([points0])
        self.points1 = np.float32([points1])

    def update(self):
        value, frame = self.cap.read()
        points0 = self.points0
        points1 = self.points1
        transform = cv.getPerspectiveTransform(points0, points1)
        frame = cv.warpPerspective(frame, transform, (np.amax(self.points1)))
        cv.imshow("frame", frame)
        key = cv.waitKey(30) & 0xff
        if key == 27:
            self.cap.release()
            output.release()
            cv.destroyAllWindows()
