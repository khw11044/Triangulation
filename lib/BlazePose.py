import cv2
import mediapipe as mp
from collections import OrderedDict

import os
from glob import glob

label_data = OrderedDict()

root = "./images"


class poseDetector():

    def __init__(self, mode= False, upBody = False, smooth = True, 
                detectionCon = 0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, 
                                     self.detectionCon, self.trackCon)

# keypoint 찾기
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # if self.results.pose_landmarks:
        #     print(self.results.pose_landmarks)
            # if draw:
            #     self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,        # keypoint를 찍음 
            #                                 self.mpPose.POSE_CONNECTIONS)           # keypoint를 연결함
        
        return img

# keypoint의 좌표
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz, cv = int(lm.x * w), int(lm.y * h), lm.z, round(lm.visibility)
    
                if id not in [1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22]:
                    lmList.append([cx,cy,cv])
                    # lmList.append([id,[cx,cy,cv]])
                    if draw:
                        if cv != 0:
                            cv2.circle(img, (lmList[-1][0], lmList[-1][1]), 5, (255,0,0), cv2.FILLED)             # keypoint 크기 및 색 
                        else :
                            cv2.circle(img, (lmList[-1][0], lmList[-1][0]), 5, (0,255,0), cv2.FILLED) 
                
                if id == 24:
                    lmList.append([int((lmList[-2][0]+cx)/2),int((lmList[-2][1]+cy)/2),round((lmList[-2][2]+cy)/2)])
                    cv2.circle(img, (lmList[-1][0], lmList[-1][1]), 5, (255,0,0), cv2.FILLED) # 중앙 골반 추가 
                    lmList.append([int((lmList[-2][0]+lmList[-3][0]+lmList[1][0]+lmList[2][0])/4),int((lmList[-2][1]+lmList[-3][1]+lmList[1][1]+lmList[2][1])/4),cv])
                    cv2.circle(img, (lmList[-1][0], lmList[-1][1]), 5, (255,0,0), cv2.FILLED)
        
        lmList.insert(1,[int((lmList[0][0]+lmList[1][0]+lmList[2][0])/3),int((lmList[0][1]+lmList[1][1]+lmList[2][1])/3),round((lmList[0][2]+lmList[1][2]+lmList[2][2])/3)])
        cv2.circle(img, (lmList[1][0], lmList[1][1]), 5, (255,0,0), cv2.FILLED)
        return lmList, img

# 0: 코, 1:neck 2: 왼쪽 어깨, 3: 오른쪽 어깨, 4: 왼쪽 팔꿈치, 5: 오른쪽 팔꿈치, 6: 왼쪽 손목, 7: 오른쪽 손목, 
# 8: 왼쪽 골반, 9: 오른쪽 골반, 10: 가운데 골반, 11: 명치, 12: 왼쪽 무릅, 13: 오른쪽 무릅, 14: 왼쪽 발목, 15: 오른쪽 발목, 
# 16: 왼쪽 발뒤꿈치, 17: 오른쪽 발뒤꿈치, 18: 왼쪽 앞꿈치, 19: 오른쪽 앞꿈치 

def main(frame):
    file_name = 'S1/Seq2/b_data/video_8/video_8_{0:04d}.png.'.format(frame)
    name = file_name.split('/')[-1]
    image = cv2.imread(file_name)
    detector = poseDetector()


    img = detector.findPose(image)
    lmList, img = detector.findPosition(img)
    # print(lmList)
    # cv2.imwrite( "pose.png",img)
    img = cv2.resize(img,dsize=(1280,1280), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.waitKey(0)
    




if __name__ == "__main__":
    img_folder = os.listdir('S1/Seq2/b_data/video_7')
    print(img_folder)
    print(len(img_folder))
    for frame in range(len(img_folder)):
        main(frame)