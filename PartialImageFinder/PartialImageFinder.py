import sys, os
import numpy as np
import zipfile
import cv2
import datetime

#Import Image to be searched for
startimage = "search.png"
small_image = cv2.imread(startimage)

#create vars
filelist = []
similarity = []
bestMatch = sys.float_info.max
counter = -1

##get all zips in subdirectories
for subdir, dirs, files in os.walk("."):
    for dir in dirs:
        path = os.path.join(subdir, dir)
        for file in os.listdir(path):
            if file.endswith(".zip"):                
                filelist.append(os.path.join(path, file))

#go through all found zips
for file in filelist:
    counter += 1
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": progress: " 
          + str(counter) + "/" + str(len(filelist)))
    zippedImgs = zipfile.ZipFile(file)

    if(bestMatch == 0.0):
        break

    #go through all images in all zips
    for img in zippedImgs.namelist():
        if(".jpg" in img or ".png" in img):
            #transform zipped files to images
            data = zippedImgs.read(img)
            nparr = np.fromstring(data, np.uint8)
            large_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            #check if image is a closer match than all previous images
            if(large_image.shape[0] < small_image.shape[0] or large_image.shape[1] < small_image.shape[1]):
                continue

            resultData = cv2.matchTemplate(large_image, small_image, cv2.TM_SQDIFF_NORMED)
            result = cv2.minMaxLoc(resultData);
            similarity.append([result, file, img])

            #Draw result rectangle
            if(result[0] < bestMatch):
                bestMatch = result[0]
                # We want the minimum squared difference
                mn,_,mnLoc,_ = cv2.minMaxLoc(resultData)
                # Draw the rectangle:
                # Step 1: Extract the coordinates of our best match
                MPx,MPy = mnLoc
                # Step 2: Get the size of the template. This is the same size as the match.
                trows,tcols = small_image.shape[:2]
                # Step 3: Draw the rectangle on large_image
                cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
                cv2.imwrite("result.png", large_image)
                print("New best Match:")
                print([result, file, img])

similarity.sort(key=lambda x: x[0][0])

print("\nTotal Best Match:")
print(similarity[0])