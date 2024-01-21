from PIL import Image
import pygame
import numpy as np
from roboflow import Roboflow

# FUNCTION THAT USES MACHINE LEARNING DETECTION TO DETECT LARGE CELESTIAL BODIES #
def getDataSpecialObjects(pathToImage):
    rf = Roboflow(api_key="CImlsMQkacpKzTkkdHaR") # we're using roboflow for the machine learning detection system

    project_nebula = rf.workspace().project("nebula-1nsta")  # initialize roboflow to detect nebulas
    model_nebula = project_nebula.version(1).model
    project_bodies = rf.workspace().project("space-image-detection") # initialize roboflow to detect large bodies (planet, suns, etc)
    model_bodies = project_bodies.version(1).model
    project_galaxy = rf.workspace().project("galaxy_detection_wcc_project") # initialize roboflow to detect galaxies
    model_galaxy = project_galaxy.version(4).model
    img_file = pathToImage # link to the image that the machine learning algorithm will detect on
    m1 = model_nebula.predict(img_file, confidence=40, overlap=30).json()
    m2 = model_bodies.predict(img_file, confidence=40, overlap=30).json()
    m3 = model_galaxy.predict(img_file, confidence=40, overlap=30).json()

    p1 = m1['predictions']
    p2 = m2['predictions']
    p3 = m3['predictions']

    special_obj = []


    a = 'y' # starting pixel column of object
    b = 'width' # width of object

    for i in p1:
        special_obj.append((i[a], i[a] + i[b])) # append the starting and ending pixel column for each special object

    for i in p2:
        special_obj.append((i[a], i[a] + i[b])) # append the starting and ending pixel column for each special object

    for i in p3:
        special_obj.append((i[a], i[a] + i[b])) # append the starting and ending pixel column for each special object

    return special_obj # returns an array of tuples containing the starting and ending pixel column for each special object
# FUNCTION DONE #


# FUNCTION THAT FINDS POSITION OF SMALL STARS USING A BFS FLOOD FILL ALGORITHM #
def getDataStars(pathToImage):
    colImage = Image.open(pathToImage)
    hPixels, vPixels = colImage.size
    threshold = 100
    bwImage = colImage.point(lambda x: 0 if x < threshold else 255)
    bwImagePixels = np.array(bwImage)
    colImagePixels = np.array(colImage)
    d = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    vis = [[False for _ in range(hPixels)] for _ in range(vPixels)]
    numStars = 0
    sizeStars = []
    locOnePixelStars = []
    avgRGBStars = []
    for i in range(vPixels):
        for j in range(hPixels):
            if vis[i][j] or bwImagePixels[i][j][0] != 255:
                continue
            vis[i][j] = True
            stack = [(i, j)]
            sizeStars.append(0)
            avgRGBStars.append([0, 0, 0])
            while stack:
                curR, curC = stack.pop()
                for rd, cd in d:
                    nextR, nextC = curR  +rd, curC + cd
                    if nextR < 0 or nextR >= vPixels or nextC < 0 or nextC >= hPixels:
                        continue
                    if vis[nextR][nextC]:
                        continue
                    if bwImagePixels[nextR][nextC][0] != 255:
                        continue
                    vis[nextR][nextC] = True
                    sizeStars[-1] += 1
                    stack.append((nextR, nextC))
                    avgRGBStars[-1][0] += colImagePixels[nextR][nextC][0]
                    avgRGBStars[-1][1] += colImagePixels[nextR][nextC][1]
                    avgRGBStars[-1][2] += colImagePixels[nextR][nextC][2]
            if 50 < sizeStars[-1] < 10000: # these two variables need to be calibrated
                numStars += 1
                locOnePixelStars.append((i, j))
                avgRGBStars[-1][0] = avgRGBStars[-1][0] // sizeStars[-1]
                avgRGBStars[-1][1] = avgRGBStars[-1][1] // sizeStars[-1]
                avgRGBStars[-1][2] = avgRGBStars[-1][2] // sizeStars[-1]
            else:
                sizeStars.pop()
                avgRGBStars.pop()
    dataStars = {}
    for i in range(numStars):
        dataStars[locOnePixelStars[i][1]] = [sizeStars[i], avgRGBStars[i]]
    return dataStars # returns an array of the pixels columns where stars exist
# FUNCTION DONE #

# FUNCTION TO FIND THE AVERAGE COLUMN BRIGHTNESS FOR EACH PIXEL -- THIS DETERMINES BACKGROUND NOSIE #
def getDataBG(pathToImage):
    image = Image.open(pathToImage) # opens the image
    hPixels, vPixels = image.size # finds the dimensions fo the image
    image = np.array(image) # puts the image into a computer usable format
    dataRGBColumn = [] # array that will contain the average column brightness for each pixel in order to determine background sound
    for i in range(hPixels):
        avgRed = 0
        avgGreen = 0
        avgBlue = 0
        for j in range(vPixels):
            avgRed += image[j][i][0]
            avgGreen += image[j][i][1]
            avgBlue += image[j][i][2]
        avgRed = avgRed // vPixels
        avgGreen = avgGreen // vPixels
        avgBlue = avgBlue // vPixels
        dataRGBColumn.append((avgRed, avgGreen, avgBlue)) # average colour is the average of averages between Red, Green, and Blue
    return dataRGBColumn
# FUNCTION DONE #

# please put the path to the photo you'd like to test the code on here
# here are the included photos: testImgNebula1.jpg, testImgNebula2.jpg, testImgNebula3.jpg, testImgNebula4.jpg, testImgStars1.jpg, testImgStars2.jpg
pathToImage = "testImgs/UsedImage5.jpg"
hPixels, vPixels = Image.open(pathToImage).size # finds the dimensions fo the image

dataRGBColumn = getDataBG(pathToImage)
dataStars = getDataStars(pathToImage) # calls the function to find stars
dataSpecialObjects = getDataSpecialObjects(pathToImage) # calls the function to find special objects

pygame.init()
pygame.mixer.init()
clock = pygame.time.Clock()

image = pygame.image.load(pathToImage)
screen = pygame.display.set_mode((hPixels, vPixels))

audio = ["audio/bg0.wav", "audio/bg1.wav", "audio/bg2.wav", "audio/bg3.wav", "audio/bg4.wav"] # these are the 4 pitches that will be the background noise. The brighter the background at current pixel column the higher the pitch

maxPitch = 4
instructions = [(0, 0)] # tuple where first item is at what column to initiate pitch and second item is what pitch
for i in range(1, hPixels):
    curBright = (dataRGBColumn[i][0] + dataRGBColumn[i][1] + dataRGBColumn[i][2]) / 3 # finds average brightness in current pixel column
    shiftedPitch = int((curBright / 255) * maxPitch) # shifts the pitch
    if shiftedPitch != instructions[-1][1]: # to avoid it replaying the background noise every second we only switch playing when the background brightness changes
        instructions.append((i, shiftedPitch))

lineColour = (255, 255, 255) # line colour (white)
lineThickness = 5 # line thickness in pixels
lineRowsStart = 0 # line begins at top
lineRowsEnd = vPixels # line ends at bottom


chime = pygame.mixer.Sound("audio/chime.wav")  # initialize pygame to play chimes for stars
chime.set_volume(0.25)  # make volume less for stars
special = pygame.mixer.Sound("audio/special1.wav")  # initialize pygame for sound for special objects
special.set_volume(1.5)

event = True
curColumn = 0
while event:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            event = False

    if instructions and instructions[0][0] == curColumn: # if we arrive at a column where the brightness changes play a different pitch for the background noise
        bg = pygame.mixer.Sound(audio[instructions[0][1]])
        bg.set_volume(1)
        instructions.pop(0)
        bg.play(-1)

    for c in dataStars:
        if curColumn == c: # if current column is at the position of a star play a noise
            chime.play()
    for cStart, cEnd in dataSpecialObjects:
        if cStart <= curColumn <= cEnd: # if current column is at the position of a special body play a noise
            special.play(-1)
            break
    else:
        special.fadeout(2000) # make the noises blend better

    screen.blit(image, (0, 0))
    pygame.draw.line(screen, lineColour, (curColumn, lineRowsStart), (curColumn, lineRowsEnd), lineThickness)
    pygame.display.update()

    curColumn += 1
    if curColumn >= hPixels:
        event = False
    clock.tick(60) # higher int value here will pass through image faster. lower int value will pass through slower

pygame.mixer.music.stop()
pygame.quit()
