from collections import defaultdict

from ultralytics import YOLO
import numpy as np
import cv2


'''
因为具备身份，使用字典，
记录


'''


'''

进一步优化，对丢失的框，进行记录速度，如果速度，在丢失的时间里面，对其框进行自动位移，对其位移后的框进行iou比对

'''



class Tracker:
    def __init__(self):
        self.model = YOLO('./best_m.pt')
        self.path = './video.mp4'
        self.post_frame = {}
        # key:id
        # value:(x1,y1,x2,y2,count)
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
            results = self.model(frame )
            results = results[0].boxes.cpu().numpy()
            new_frame = {}
            for result in results:
                conf = result.conf
                if conf>0.5:
                    new_box = result.data[:, :4]
                    new_box = np.squeeze(new_box)
                    x1, y1, x2, y2 = map(int, new_box)
                    max_iou = 0.
                    temp_id = None
                    for old_id,( x1_, y1_, x2_, y2_,  miss_count) in self.post_frame.items():

                        old_box = (x1_, y1_, x2_, y2_,)
                        iou = self.iou(new_box, old_box)
                        if iou > max_iou:
                            max_iou = iou
                            temp_id = old_id

                    if max_iou > 0.3 and temp_id is not None:
                        match_id = temp_id
                    else:
                        miss_count = 0
                        match_id = self.id
                        self.id += 1
                    new_frame[match_id]=[x1, y1, x2, y2, miss_count]
                    if miss_count <30:

                        self.post_frame[match_id] = new_frame[match_id]
                # 因为置信度低所以才会丢失目标，所以才需要对低置信度进行处理
                else:
                    new_box = result.data[:, :4]
                    new_box = np.squeeze(new_box)
                    x1, y1, x2, y2 = map(int, new_box)
                    max_iou = 0.
                    for old_id, (x1_, y1_, x2_, y2_, miss_count) in self.post_frame.items():
                        old_box = (x1_, y1_, x2_, y2_,)
                        iou = self.iou(new_box, old_box)
                        temp_id = None
                        if iou > max_iou:
                            max_iou = iou
                            temp_id = old_id

                    if max_iou > 0.3 and temp_id is not None:
                        match_id = temp_id
                    new_frame[match_id] = [x1, y1, x2, y2, miss_count]
                    for key in self.post_frame.keys():
                        if key not in new_frame.keys():
                            self.post_frame[key][4] += 1
                x1, y1, x2, y2, miss_count = new_frame[match_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (244, 122, 233), 2)
                cv2.putText(frame, f'id{match_id} ', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (123, 123, 232),
                            1, )
            if miss_count < 30:
                self.post_frame[match_id] = new_frame[match_id]
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    tracker = Tracker()
    tracker.main()