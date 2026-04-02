import numpy as np
from ultralytics import YOLO
import cv2

'''
使用模型得到对应图片框的坐标
永远只需要比对这一帧和上一帧
    存储第一帧图片的所有框的信息，并且进行编号
    对下一帧图片进行与上一帧的图片的所有iou比对，对于不存在的编号，加载作为下一帧的对比

每次都生成新的new_frame的信息，将new_frame与old_frame比对，覆盖掉old_frame,进行图片展示


'''


class Tracker():
    def __init__(self):
        self.model = YOLO('./best_m.pt')

        video_path = './video.mp4'
        self.cap = cv2.VideoCapture(video_path)
        self.post_frame = []
        self.id = 0

    def main(self):
        while self.cap.isOpened():

            ret, frame = self.cap.read()
            if not ret:
                break
            result = self.model(frame, conf=0.5)
            # 存储这一帧的所有坐标信息
            result = result[0].cpu().numpy()
            boxes = result.boxes

            new_frame = []
            for box in boxes:
                data = box.data
                new_box = data[:,:4]
                new_box = np.squeeze(new_box)
                x1 , y1 , x2 , y2 = map(int , new_box)
            #     data:(坐标，置信度，类别)
                max_iou = 0
                for old_id , old_data in self.post_frame:
                    iou = self.iou(old_data[:4] , new_box)
                    if iou > max_iou:
                        max_iou = iou
                        temp_id = old_id

                if max_iou > 0.3:
                    match_id = temp_id
                else:
                    match_id = self.id
                    self.id += 1
                new_frame.append((match_id , new_box))
                cv2.rectangle(frame , (x1 , y1),( x2 , y2),(144,122,244) , 2)
                cv2.putText(frame , f'id:{match_id}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )
            self.post_frame = new_frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break





    def iou(seld, box1, box2):
        area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        lx = np.maximum(box1[0], box2[0])
        ly = np.maximum(box1[1], box1[1])
        rx = np.minimum(box1[2], box2[2])
        ry = np.minimum(box1[3], box2[3])
        inter_area = np.maximum((rx - lx), np.array(0)) * np.maximum((ry - ly), np.array(0))
        iou = inter_area / (area_1 + area_2 - inter_area)
        return iou

if __name__ == '__main__':
    tracker = Tracker()
    tracker.main()
