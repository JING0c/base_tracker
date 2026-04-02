import cv2
import numpy as np
from ultralytics import YOLO

'''
问题：
模型在丢失目标后，在重新出现目标后，他会将id增加，而不是之前的目标了
优化思路：
如果目标不是在图像边缘丢失，而目标突然在图像中间丢失，就说明目标还存在，就保留数字id ,z
但是出现了新的问题，就是怎么能让计算机知道还是那个物体，并且将id赋值给他
拿到模型输出的特征，进行模型的特征比对，如果特征的余弦相似度高，就将存储的id赋值给他

那要怎么解决框突然消失呢，让框继续走呢：
但是我个人觉得完全没必要让框一直存在，如果存在框不是在屏幕边缘消失的，而久久不能出现，就将信息打印出来，人为检测，我只要保证那个物体是在屏幕边缘消失的就可以了
或者说我还有猜想，就是对模型进行更换，使用transforms的多头自注意力机制一样，像预测单词一样，预测行为轨迹，但是还是需要很多细化的东西


'''





'''
只比对这一帧和上一帧
进行讲新的box与旧的box进行比对，若出现存在的就讲旧的id赋值给新的id
若出现不存在的，就进行新加id

'''


'''

选择优化策略：
将self.posty_frame进行更改为（id ， box , missed_count）
    miss_count :默认为零
                如果一个框消失了一帧就记录miss_count + 1
                如果miss_coun 达到30 就剔除掉
进行高低分筛选（
对高分，赋予id
对低分：与消失的特征库做比对，比对上了就重新拉回
）



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
            results = self.model(frame )
            results = results[0].boxes.cpu().numpy()
            for result in results:
                new_frame= []
                conf = result.conf
                '''
                              对于高于0.7的，模型已经确认为识别出来的物体
                              对于旧的进行赋值，对于新的进行安排编号
                              在一个新的box与每个旧的box比对时，如果出现就的没有被赋值的框就将其对应的吗miss_count +1
                              如果说一个miss_count的数达到了30，那就将他从self.post_frame里面剔除掉
    
                              '''
                if conf > 0.7:

                    new_box = result.data[:, :4]
                    new_box = np.squeeze(new_box)
                    x1, y1, x2, y2 = map(int, new_box)
                    max_iou = 0.

                    for old_id , old_box , miss_count in self.post_frame:
                        iou = self.iou(new_box , old_box)
                        if iou>max_iou:
                            max_iou = iou
                            temp_id = old_id

                    if max_iou > 0.3:
                        match_id = temp_id
                    else:
                        miss_count = 0
                        match_id = self.id
                        self.id +=1
                    new_frame.append((match_id , (x1, y1, x2, y2) , miss_count))
                    new_frame = self.miss_num(new_frame , self.post_frame)
            else:
                '''
                低置信度的，将其与过去存的30帧的图片里面所有的进行比对，对于iou高于0.3的进行赋予旧值，
                如果匹配上了，就对其miss_count就行修改为0
                
                '''
                new_box = result.data[:, :4]
                new_box = np.squeeze(new_box)
                x1, y1, x2, y2 = map(int, new_box)
                max_iou = 0.
                for old_id , old_box , miss_count in self.post_frame:
                    iou = self.iou(new_box, old_box)
                    if iou > max_iou:
                        max_iou = iou
                        temp_id = old_id
                if max_iou > 0.3:
                    match_id = temp_id
                    miss_count = 0
                new_frame.append((match_id, (x1, y1, x2, y2), miss_count))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (244, 122, 233), 2)
                cv2.putText(frame, f'id{match_id} ', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (123, 123, 232), 1, )
            self.post_frame.append(new_frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break





    def miss_num(self , new_frame , old_frame):
        match_id, _, miss_count = new_frame[0]
        if miss_count not in old_frame:
            miss_count += 1
        new_frame = [(match_id, _, miss_count)]
        idx = miss_count==30
        part1 = new_frame[0][:idx]
        part2 = new_frame[0][idx:]
        new_frame = part1.extend(part2)
        return new_frame







if __name__ == '__main__':
    tracker = Tracker()
    tracker.main()