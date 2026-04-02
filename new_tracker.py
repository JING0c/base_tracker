import cv2
import numpy as np
from ultralytics import YOLO

'''
只比对这一帧和上一帧
进行讲新的box与旧的box进行比对，若出现存在的就讲旧的id赋值给新的id
若出现不存在的，就进行新加id

'''


class Tracker:
    def __init__(self):
        self.model = YOLO('./best_m.pt')
        self.path = './video.mp4'
        self.post_frame = []
        self.id = 0

    def iou(self, box1, box2):
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        lx = np.maximum(box1[0], box2[0])
        ly = np.maximum(box1[1], box2[1])
        rx = np.minimum(box1[2], box2[2])
        ry = np.minimum(box1[3], box2[3])

        inter_iou = np.maximum((rx-lx) , np.array(0))*np.maximum((ry-ly),np.array(0))

        iou = inter_iou/(area1+area2-inter_iou)

        return iou

    def main(self):
        cap = cv2.VideoCapture(self.path)

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            results = self.model(frame , conf=0.5)
            results = results[0].boxes.cpu().numpy()
            new_frame = []
            for result in results:
                new_box = result.data[:,:4]
                new_box = np.squeeze(new_box)
                x1 , y1 , x2 , y2 = map(int,new_box)
                max_iou = 0.
                for old_id , old_box in self.post_frame:
                    iou = self.iou(new_box,old_box)
                    if iou > max_iou:
                        max_iou = iou
                        temp_id = old_id

                if max_iou > 0.3:
                    match_id = temp_id
                else:
                    match_id = self.id
                    self.id += 1
                new_frame.append((match_id,new_box))
                cv2.rectangle(frame , (x1,y1) , (x2,y2),(244,122,233),2)
                cv2.putText(frame , f'id{match_id} ',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6 ,(123,123,232),1,)
            self.post_frame = new_frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    tracker = Tracker()
    tracker.main()