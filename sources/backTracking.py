print("\t\t\t##Running ","capture-test.py##");
try:
    print("Importing Packages Required...",end="##")
    import cv2
    import numpy as np
    #import gesture_verify as gvf
    from collections import deque
    #import tensorflow as tf
    print("...Import Sucessful")
except Exception as exc:
    print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")
    print("Error is : \n\t",exc)
print("Setting Global Variables...",end=' ')
#gesture_model=gvf.ModelLoader()
#skin_model  = svf.ModelLoader()
#import pickle
#with open('Originals/Models/RF.pkl', 'rb') as f:
#    other_model = pickle.load(f)
#    other_model.verbose = False
#print("=========Other Model Loaded.===========\n")

names = ['1','2','3','4','5','Fist','None','OK','Palm']
currentframe=1
i=0
bg=None
flag1=0
kernel = np.ones((2,2),np.uint8) #kernel for Opening and Closing morphologyEx
pts = deque(maxlen=32)
counter = 0
(dX, dY) = (0, 0)
direction = ""

cascade=cv2.CascadeClassifier('C:/Users/nabin/Desktop/palm.xml')
print("## Done.")


def average(img):
    global bg
    if bg is None :
        bg = img.copy().astype("float")
        print("## Running Initial Background Scan . . . ")
        return
    cv2.accumulateWeighted(img,bg,0.3)
#weight(0.3 now) is update bg speed(how fast the accumulator “forgets” about earlier images)
    if i==49:
        print(" Scan Done...Ready For Threshold ##")

def tracker(c,frame):
    global i
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if i==50:
        for z in range (1,33):
            pts.append(center)
        i+=1
    #print(center)
    if radius > 10:
        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        #print('The point is ',pts)
    for index in np.arange(1, len(pts)):
        #print("here")
        # if either of the tracked points are None, ignore
        # them
        if pts[index - 1] is None or pts[index] is None:
                continue
        if index == 1 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-10][0] - pts[index][0]
            dY = pts[-10][1] - pts[index][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"

            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                    direction = dirX if dirX != "" else dirY
            #print(dirX,dirY)

        thickness = int(np.sqrt(32/ float(index + 1)) * 2)
        cv2.line(frame, pts[index - 1], pts[index], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (150,frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 1)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

def segmenter(img):
    global bg
    #cv2.accumulateWeighted(img,bg,0.001)
    diff = cv2.absdiff(bg.astype("uint8"),img)
    #th1 = cv2.threshold(diff,30,255,cv2.THRESH_BINARY) [1]
    #cv2.imshow("Binary only",th1)
    thres = cv2.threshold(diff,30,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)      #Opening
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8) )     #Closing
    cnts,_ = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segment = max(cnts, key=cv2.contourArea)
        return (thres,segment)

if __name__ == "__main__":
    print("\n\t\t** Hello from Main of 'capture-test.py' **")
    histTaken = False
    print("Opening Camera...")
    camera =cv2.VideoCapture(0)
    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100) #     smoothing filter
        frame = cv2.flip(frame, 1)
        font=cv2.FONT_HERSHEY_SIMPLEX
        crop = frame[90:400,370:630]

        #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)     ####### Gaussian Blur ###########

        keypress = cv2.waitKey(2) & 0xFF
        if keypress is not 0xFF:
            print("KeyPressed!! : ",keypress)
            if keypress == ord("c"):
                name = 'DATA/Capture' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name,crop)
                currentframe += 1
            if keypress == ord("p"):
                flag1=1
            if keypress == ord("o") or currentframe==500:
                flag1=0
            if keypress == ord("r"):
                histTaken = False
                #currentframe=1
            if flag1==1:
                name = 'DATA/ContCapture' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name,crop)
                currentframe += 1

        if i<50:
            #average(gray)
            i=i+1
        else:
            sg_return=[1,1]
            #sg_return = segmenter( gray )
            if sg_return is not None:
                #faces = cascade.detectMultiScale(crop, 1.2, 5)
                if histTaken == False:
                    cv2.rectangle(frame,(450,190),(550,380),(255,0,0),0) # Palm
                    cv2.rectangle(frame,(400,220),(450,350),(255,0,0),0) # Thumb
                    cv2.putText(frame,'Palm Box',(430,180),font,1,(0,70,250),2,cv2.LINE_AA)
                    if keypress == ord("h"):
                        palmFrame = frame[195:375,455:545]
                        cv2.imshow('PalmBox',palmFrame)
                        #cv2.imwrite('PalmImage1.jpg',palmFrame)
                        palmHSV = cv2.cvtColor(palmFrame,cv2.COLOR_BGR2HSV)
                        palmHist = cv2.calcHist([palmHSV],[0, 1],None,[180, 256],[0,180,0,256])
                        cv2.normalize(palmHist,palmHist,0,255,cv2.NORM_MINMAX)
                        histTaken = True
                if histTaken == True:
                    cropHSV=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([cropHSV],[0,1],palmHist,[0,180,0,256],1)
                    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                    cv2.filter2D(dst,-1,disc,dst)
                    # threshold and binary AND
                    _,thresh = cv2.threshold(dst,50,255,0)
                    thresh = cv2.merge((thresh,thresh,thresh))
                    res = cv2.bitwise_and(crop,thresh)
                    res = np.vstack((thresh,res))

                    cv2.imshow('RESULT :: Thres and BitwiseOutput',res)
                    #cv2.imshow("PALM",palm1)



        cv2.rectangle(frame,(370,90),(630,400),(255,0,0),0)
        cv2.putText(frame,'Region of Interest',(350,80),font,1,(0,70,250),2,cv2.LINE_AA)
        cv2.imshow('Live Feed',frame)
        #cv2.imshow('Cropped',crop)


        if keypress == ord("q"):
            print("Exit Key Pressed")
            camera.release()
            cv2.destroyAllWindows()
            break
        if keypress == ord("b"):
            print("   ## Running Background Scan...",end=' ')
            cv2.accumulateWeighted(gray,bg,0.4)
            print(" .. DONE  ##")

        if keypress == ord("x"):
            name = 'DATA/BlackCapture' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name,thres)
            currentframe += 1





print("Done")
