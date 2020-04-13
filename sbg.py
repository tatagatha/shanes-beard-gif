import argparse
import cv2
import glob
import os
import numpy as np

# Based heavily on
# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

# detect location of biggest face
# return image and it's x/y location relative to the root


def detect_face(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        face_frame = img[y:y + h, x:x + w]
        return (face_frame, x, y)


def detect_eyes(face, cascade):
    img, fX, fY = face
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    left_eye = None
    right_eye = None
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = (img[y:y + h, x:x + w], x + fX, y + fY)
        else:
            right_eye = (img[y:y + h, x:x + w], x + fX, y + fY)
    return left_eye, right_eye


# reduce the area of the eye image by some reasonable amount to cut out noise
# or by unreasonable amount
# This feels like cheating on making the blob detctor work
mysteryEyebrowFactor = .2
def narrowEyes(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 3.5)
    # img = img[eyebrow_h:height-eyebrow_h, int(width*.2):int(width*.8)]  # cut eyebrows
    img = img[eyebrow_h:height - eyebrow_h, int(width * mysteryEyebrowFactor):int(
        width * (1 - mysteryEyebrowFactor))]  # cut eyebrows
    return img

# dumb function writting, can probably write better or learn pass by object reference  better
# but like, you learned python today so don't worry
# returns what is cut off the top and left only
def estimateNarrowEyesOffset(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 3.5)
    return (eyebrow_h, int(width * mysteryEyebrowFactor))

def convertEye(img, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return img

#Global detector used by blob_process
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByColor = True
detector_params.blobColor = 0
detector = cv2.SimpleBlobDetector_create(detector_params)

# Super ineffiecient loop that checks many different thresholds
# and then grabs the largest keypoint and returns it's point location
def blob_process(img):
    keypoints = []
    # try each threshold within reason
    for threshold in range(2, 12):
        keypoints.extend(testEye(img, threshold * 8, detector))
        keypoints = list(dict.fromkeys(keypoints))
    if not keypoints:
        return None
    currentKeyPoint = None
    # get the biggest keypoint. We think it will be the pupil
    for keypoint in keypoints:
        if currentKeyPoint is None:
            currentKeyPoint = keypoint
            continue
        if keypoint.size > currentKeyPoint.size:
            currentKeyPoint = keypoint
    return currentKeyPoint

# I fussed with this a lot to make some iterations that worked for my conditions
def testEye(img, threshold, detector):
    img = convertEye(img, threshold)
    img = cv2.erode(img, None, iterations=4)  # 1
    img = cv2.dilate(img, None, iterations=6)  # 2
    # Original example had less blur. I added more for the large scale image.
    # Could scale the number to image size of the eye chunk
    img = cv2.blur(img, (20, 20))  # 3
    keypoints = detector.detect(img)
    return keypoints

# crop the image maintain 3/4 ratio
# 4 pd WIDE * 3/4 = 3
# 5.333333333 pd tall * 3/4 = 4
# center on the face
# 2.333333 above 3 below pupil line
def cropImage(img, pupilPoints):
    height, width = img.shape[:2]
    # pray there is two keypoints...
    leftPupil, rightPupil = pupilPoints
    lx, ly = leftPupil
    rx, ry = rightPupil
    pupilDistance = rx - lx
    pupilCenter = lx + pupilDistance / 2
    pupilHeight = ly

    startX = int(pupilCenter - (2 * pupilDistance))
    endX = int(pupilCenter + (2 * pupilDistance))

    startY = int(pupilHeight - ((2 + (1 / 3)) * pupilDistance))
    endY = int(pupilHeight + (3 * pupilDistance))

    # pupil Height line
    if drawDebug:
        phlStart = (0, int(pupilHeight))
        phlEnd = (width, int(pupilHeight))
        cv2.line(img, phlStart, phlEnd, [0, 255, 0], 1)

    # pupil Center line
    if drawDebug:
        pclStart = (int(pupilCenter), 0)
        pclEnd = (int(pupilCenter), height)
        cv2.line(img, pclStart, pclEnd, [255, 0, 0], 1)

    # crop box
    if drawDebug:
        cv2.rectangle(img, (startX, startY), (endX, endY), [0, 0, 255], 5)
    img = img[startY:endY, startX:endX]
    return img

#Global classifiers used by process image and below
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
globalCount = 1
def processImage(imgPath, outputPath):
    global globalCount
    img = cv2.imread(imgPath)
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    face = detect_face(img, face_cascade)
    pupilPoints = []
    if face is not None:
        eyes = detect_eyes(face, eye_cascade)
        for eyeInfo in eyes:
            if eyeInfo is None:
                print("   Problem: eye info none, probably only one eye found")
            else:
                eye, eX, eY = eyeInfo
                if eye is not None:
                    # Aaahhh we chopped a bunch here? how much?
                    narrowOffset = estimateNarrowEyesOffset(eye)
                    offsetX, offsetY = narrowOffset
                    eye = narrowEyes(eye)
                    pupilPoint = blob_process(eye)
                    if pupilPoint is None:
                        print("   Problem: no pupil seen")
                    else:
                        # where is that point compared to the whole image?
                        px, py = pupilPoint.pt
                        adjustedPoint = (px + eX + offsetX, py + eY + offsetY)
                        pupilPoints.append(adjustedPoint)
                        if drawDebug:
                            keydraw = cv2.drawKeypoints(
                                eye, [pupilPoint], eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(pupilPoints) is 1:
        print("   Problem: Only one pupil seen")
    if len(pupilPoints) > 1:
        # TODO: Rotate if necessary by looking at angle between eyes.
        # Fortunately Shane keeps his head straight.
        # Crop the image around pupil points
        img = cropImage(img, pupilPoints)
        # scale
        img = cv2.resize(img, (outputWidth, outputHeight), interpolation=cv2.INTER_AREA)
        #outputFilePath = "./output/" + imgPath[imgPath.rfind("/"):]
        outputFilePath = os.path.join(outputPath , str(globalCount) + ".jpg")
        print("Writing file: ", outputFilePath)
        cv2.imwrite(outputFilePath, img)
        globalCount = globalCount + 1

def makeOutputDirectory(outputPath):
    if not os.path.exists(outputPath):
        print("Making output directory: ", outputPath)
        os.mkdir(outputPath)

def getInputFiles(inputPath):
    path = os.path.join('.', inputPath)
    return os.listdir(path)

# help. Now you are putting lipstick on this pig.
parser = argparse.ArgumentParser(description='Process Shane Beard Files')
parser.add_argument('-i','--input-path', default="./input", help="Input Directory")
parser.add_argument('-o','--output-path', default="./output", help="Output destination")
parser.add_argument('--output-height', type=int, default=400, help="Final height of files")
parser.add_argument('--output-width', type=int, default=300, help="Final width of files")
parser.add_argument("--draw-debug-lines", action='store_true', help="Draw debug lines.")

# Configurations
drawDebug = False
outputHeight = 800
outputWidth = 600

def main():
    global outputHeight
    global outputWidth
    global drawDebug
    args = parser.parse_args()
    #print(args)
    outputPath = args.output_path
    outputHeight = args.output_height
    outputWidth = args.output_width
    drawDebug = args.draw_debug_lines
    inputPath = args.input_path

    # make sure we have an output directory
    makeOutputDirectory(outputPath)

    files = getInputFiles(inputPath)
    files.sort()
    for file in files:
        path = os.path.join(inputPath, file)
        print(path)
        processImage(path, outputPath)
    # there used to be so many debugging images in this.
    cv2.destroyAllWindows()

main()
