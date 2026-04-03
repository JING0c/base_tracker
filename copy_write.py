from ultralytics import YOLO
import cv2
import numpy as np

'''
架构：
    1分为高分框和低分框
    
    
    2 处理高分框
    
    
    3 处理低分框
    
    4 画图保存视频

'''


class MultiROICapturer:
    def __init__(self):
        self.all_rois = []  # 存放所有已经画好的多边形
        self.current_pts = []  # 存放当前正在画的这个多边形的点

    def draw_roi(self, event, x, y, flags, param):
        # 左键点击：添加点
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
        # 右键点击：闭合当前多边形，存入总列表，并清空画笔准备画下一个
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_pts) >= 3:
                self.all_rois.append(np.array(self.current_pts, np.int32))
                self.current_pts = []  # 清空，准备画下一个区域
                print(f"✅ 成功划定第 {len(self.all_rois)} 个区域！可以继续用左键画下一个，或按回车开始。")
            else:
                print("⚠️ 一个区域至少需要 3 个点才能闭合哦！")

    def get_rois_from_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("读取视频失败！")
            return []

        cv2.namedWindow('Select Multiple ROIs', cv2.WINDOW_NORMAL)  # 允许调整窗口大小
        cv2.setMouseCallback('Select Multiple ROIs', self.draw_roi)

        print("👇 【多区域划分操作指南】 👇")
        print("1. 🖱️ 鼠标【左键】依次点击，勾勒区域轮廓。")
        print("2. 🖱️ 鼠标【右键】点击，闭合当前区域（颜色会变实）。然后可以直接去画下一个！")
        print("3. ⌨️ 按下【Enter(回车键)】完成所有区域划分，正式启动追踪！")

        while True:
            temp_frame = frame.copy()

            # 画出已经完成的区域 (用蓝色)
            for i, roi in enumerate(self.all_rois):
                cv2.polylines(temp_frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)
                # 给画好的区域标个号，方便你认
                cv2.putText(temp_frame, f"Area {i + 1}", tuple(roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 实时画出当前正在画的区域 (用绿色)
            if len(self.current_pts) > 0:
                cv2.polylines(temp_frame, [np.array(self.current_pts)], isClosed=False, color=(0, 255, 0), thickness=2)
                for pt in self.current_pts:
                    cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow('Select Multiple ROIs', temp_frame)

            if cv2.waitKey(1) & 0xFF == 13:  # 13 是回车键
                # 如果用户画了一半没按右键就按了回车，帮他自动闭合收尾
                if len(self.current_pts) >= 3:
                    self.all_rois.append(np.array(self.current_pts, np.int32))
                break

        cv2.destroyWindow('Select Multiple ROIs')
        cap.release()

        return self.all_rois  # 返回的是一个列表，里面装了多个多边形矩阵


class IouTracker:
    def __init__(self, path='./video.mp4'):
        self.path = path
        self.model = YOLO('./best_m.pt')
        self.post_frame = {}
        self.id = 0.
        self.save_id = []


        pass

    def iou(self, box1, box2):
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        lx = np.maximum(box1[0], box2[0])
        ly = np.maximum(box1[1], box2[1])
        rx = np.minimum(box1[2], box2[2])
        ry = np.minimum(box1[3], box2[3])
        inter_iou = np.maximum((rx - lx), 0) * np.maximum((ry - ly), 0)
        # 别忘了分母加 1e-6 防止除以 0 报错
        iou = inter_iou / (area1 + area2 - inter_iou + 1e-6)
        return iou

    def main(self):
        # =======================选取目标区域进行识别============================
        capturer = MultiROICapturer()
        self.roi_list = capturer.get_rois_from_first_frame(self.path)

        if len(self.roi_list) == 0:
            print("没有画任何有效区域，程序退出！")
            return
        # =======================抓捕一次函数的起点和中电=========================
        line_capture = LineCapturer()
        self.line_point = line_capture.get_lines(self.path)
        #self.line_point ：list[((p1),(p2))]
        # ======================调用一次函数表达式==========================
        line_function = LineFunction()
        self.line_functions = line_function.get_lines(self.line_point)

        cap = cv2.VideoCapture(self.path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_result.mp4', fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.model(frame, verbose=False)
            boxes_data = results[0].boxes.data.cpu().numpy()

            now_match_id = []
            high_score = []
            low_score = []
            new_frame = {}

            # ======================= 过滤结界：分为高分和低分 ================================
            for data in boxes_data:
                conf = data[4]
                box = data[:4]

                curr_cx = (box[0] + box[2]) / 2.0
                curr_cy = (box[1] + box[3]) / 2.0

                # 🔪 核心升级：遍历所有区域
                in_any_roi = False
                for roi_pts in self.roi_list:
                    # 只要中心点在其中任意一个区域内 (>=0)，就标记为 True 并跳出检查
                    if cv2.pointPolygonTest(roi_pts, (curr_cx, curr_cy), False) >= 0:
                        in_any_roi = True
                        break

                        # 如果它不在任何一个你画的区域里，直接扔掉
                if  in_any_roi:
                    continue

                    # 只有合法的车，才进入高低分篮子
                if conf > 0.5:
                    high_score.append(box)
                elif conf > 0.1:
                    low_score.append(box)
            #             ===================处理高分框=======================
            for new_box in high_score:
                max_iou = 0
                temp_id = None
                new_box = np.squeeze(new_box)
                new_x1, new_y1, new_x2, new_y2 = new_box[:4]
                old_vx_glob = None
                old_vy_glob = None
                miss_count_glob = None
                for old_id, (old_x1, old_y1, old_x2, old_y2, miss_count, old_vx,
                             old_vy) in self.post_frame.items():
                    old_box = [old_x1 + old_vx, old_y1 + old_vy, old_x2 + old_vx, old_y2 + old_vy]
                    iou = self.iou(new_box, old_box)
                    old_vx_glob = old_vx
                    old_vy_glob = old_vy
                    miss_count_glob = miss_count
                    if iou > max_iou:
                        max_iou = iou
                        temp_id = old_id

                if max_iou > 0.3 and temp_id is not None:
                    match_id = temp_id
                    old_data = self.post_frame[match_id]
                    o_x1, o_y1, o_x2, o_y2, o_miss, o_vx, o_vy = old_data

                    # 🌟 绝招：计算新框和旧框的中心点
                    new_cx = (new_x1 + new_x2) / 2.0
                    new_cy = (new_y1 + new_y2) / 2.0
                    old_cx = (o_x1 + o_x2) / 2.0
                    old_cy = (o_y1 + o_y2) / 2.0

                    frame_interval = o_miss + 1.0

                    # 🌟 用中心点的位移来算真实速度！
                    curr_vx = (new_cx - old_cx) / frame_interval
                    curr_vy = (new_cy - old_cy) / frame_interval

                    # 因为质心速度非常稳定，我们可以用 0.8 的权重信任当前速度，让红框能迅速跟上黑车！
                    smooth_vx = 0.2 * o_vx + 0.8 * curr_vx
                    smooth_vy = 0.2 * o_vy + 0.8 * curr_vy

                    # 把死区调到极小（只过滤绝对的静止，比如 1 个像素）
                    if abs(smooth_vx) < 1.0: smooth_vx = 0.0
                    if abs(smooth_vy) < 1.0: smooth_vy = 0.0

                    calculate = CalculateM()
                    result = calculate.run_m(self.line_functions ,new_cx  , new_cy , smooth_vx , smooth_vy)
                    if result  and match_id not in self.save_id:
                        self.save_id.append(match_id)

                    new_frame[match_id] = [new_x1, new_y1, new_x2, new_y2, 0, smooth_vx, smooth_vy]
                    now_match_id.append(match_id)
                    pass
                else:
                    # 处理新的数据框
                    match_id = self.id
                    self.id += 1
                    new_frame[match_id] = [new_x1, new_y1, new_x2, new_y2, 0, 0., 0.]
                    now_match_id.append(match_id)

            #             =====================处理低分框=========================
            for new_box in low_score:
                max_iou = 0
                temp_id = None
                new_box = np.squeeze(new_box)
                new_x1, new_y1, new_x2, new_y2 = new_box[:4]
                old_vx_glob = None
                old_vy_glob = None
                miss_count_glob = None
                for old_id, (old_x1, old_y1, old_x2, old_y2, miss_count, old_vx,
                             old_vy) in self.post_frame.items():
                    old_box = [old_x1 + old_vx, old_y1 + old_vy, old_x2 + old_vx, old_y2 + old_vy]
                    iou = self.iou(new_box, old_box)
                    old_vx_glob = old_vx
                    old_vy_glob = old_vy
                    miss_count_glob = miss_count
                    if iou > max_iou:
                        max_iou = iou
                        temp_id = old_id
                if max_iou > 0.3 and temp_id is not None:
                    match_id = temp_id
                    old_data = self.post_frame[match_id]
                    o_x1, o_y1, o_x2, o_y2, o_miss, o_vx, o_vy = old_data

                    # 🌟 绝招：计算新框和旧框的中心点
                    new_cx = (new_x1 + new_x2) / 2.0
                    new_cy = (new_y1 + new_y2) / 2.0
                    old_cx = (o_x1 + o_x2) / 2.0
                    old_cy = (o_y1 + o_y2) / 2.0

                    frame_interval = o_miss + 1.0

                    # 🌟 用中心点的位移来算真实速度！
                    curr_vx = (new_cx - old_cx) / frame_interval
                    curr_vy = (new_cy - old_cy) / frame_interval

                    # 因为质心速度非常稳定，我们可以用 0.8 的权重信任当前速度，让红框能迅速跟上黑车！
                    smooth_vx = 0.2 * o_vx + 0.8 * curr_vx
                    smooth_vy = 0.2 * o_vy + 0.8 * curr_vy

                    # 把死区调到极小（只过滤绝对的静止，比如 1 个像素）
                    if abs(smooth_vx) < 1.5: smooth_vx = 0.0
                    if abs(smooth_vy) < 1.5: smooth_vy = 0.0
                    calculate = CalculateM()
                    result = calculate.run_m(self.line_functions , new_cx, new_cy, smooth_vx, smooth_vy)
                    if result and match_id not in self.save_id:
                        self.save_id.append(match_id)



                    new_frame[match_id] = [new_x1, new_y1, new_x2, new_y2, 0, smooth_vx, smooth_vy]
                    now_match_id.append(match_id)

                    #                         ===========================处理消失的目标=======================
            for idx, data in self.post_frame.items():
                if idx not in now_match_id:
                    old_x1, old_y1, old_x2, old_y2, miss_count, old_vx, old_vy = data
                    new_miss_count = miss_count + 1
                    if new_miss_count < 30:
                        up_x1 = old_x1 + old_vx
                        up_y1 = old_y1 + old_vy
                        up_x2 = old_x2 + old_vx
                        up_y2 = old_y2 + old_vy

                        new_frame[idx] = [up_x1, up_y1, up_x2, up_y2, new_miss_count, old_vx, old_vy]

            # for match_id, data in new_frame.items():
            #     t_x1, t_y1, t_x2, t_y2, t_miss, t_vx, t_vy = data
            #     # 如果框的高度大于 200，或者速度大得离谱
            #     if t_y2 - t_y1 > 200 or abs(t_vy) > 50:
            #         print("\n" + "=" * 40)
            #         print(f"🚨 抓到异常！嫌疑犯 ID: {match_id}")
            #         print(f"当前坐标: y1={t_y1}, y2={t_y2} (高度={t_y2 - t_y1})")
            #         print(f"当前速度: vx={t_vx}, vy={t_vy}")
            #         print(f"丢失帧数: {t_miss}")
            #         print("=" * 40)
            #
            #         # 关键绝招：让视频在这里死死停住！直到你按任意键才继续下一帧
            #         cv2.waitKey(0)

            #                   =============图像覆盖================

            self.post_frame = new_frame
            # 😼画出统计的线和数字
            for p1 , p2 in self.line_point:
                cv2.line(frame , p1 , p2 , (0,0,255) , 3)
                cv2.putText(frame , f'total_count:{len(self.save_id)}',(50,50) ,cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 122), 2 )

            # 🌟 画出你圈定的所有结界
            for i, roi_pts in enumerate(self.roi_list):
                # 不同的区域可以用同一个颜色，也可以根据 index 给不同颜色
                cv2.polylines(frame, [roi_pts], isClosed=True, color=(255, 150, 0), thickness=2)
                cv2.putText(frame, f"ROI {i + 1}", tuple(roi_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)

            for target_id, target_data in self.post_frame.items():
                t_x1, t_y1, t_x2, t_y2, t_miss, _, _ = map(int, target_data)

                if t_miss == 0:
                    color = (244, 122, 233)
                    text = f'id:{target_id}'
                else:
                    color = (0, 0, 255)
                    text = f'id:{target_id} (Pred)'

                cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), color, 2)
                cv2.putText(frame, text, (t_x1, t_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        out.release()  # 退出时才关机
        cv2.destroyAllWindows()
        print(f'😼统计的数量为：{len(self.save_id)}')

# ===============================车流统计函数==================================
    '''
    需要速度和质心位置
    定义函数位置
    '''

class LineCapturer:
    def __init__(self):
        self.lines = []  # 存最终的线段：[((x1,y1), (x2,y2)), ...]
        self.current_start = None
        self.drawing = False
        self.temp_end = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # 记录一对点
                self.lines.append((self.current_start, (x, y)))
                self.drawing = False
                self.current_start = None

    def get_lines(self, video_path):
        # 🌟 这里就是你说的：必须传 video_path 才能读到画面！
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret: return []

        cv2.namedWindow('Draw Lines')
        cv2.setMouseCallback('Draw Lines', self._mouse_callback)

        print("👆 鼠标左键点击并拖动来画直线，画完按回车键确认。")

        while True:
            img_copy = frame.copy()
            # 画已经固定下来的线
            for p1, p2 in self.lines:
                cv2.line(img_copy, p1, p2, (0, 255, 0), 2)
            # 画正在拖动的预览线（橡皮筋效果）
            if self.drawing and self.current_start and self.temp_end:
                cv2.line(img_copy, self.current_start, self.temp_end, (0, 255, 255), 1)

            cv2.imshow('Draw Lines', img_copy)
            if cv2.waitKey(1) & 0xFF == 13:  # 回车退出
                break

        cv2.destroyWindow('Draw Lines')
        return self.lines  # 🌟 这里才真正把点集交还给你的 main

class LineFunction:
    def __init__(self,):
        self.lines = []
        pass
    def get_lines(self , data ,):

        for p1 , p2 in data:
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0]*p1[1] - p1[0]*p2[1]
            lis= [A , B ,C]
            self.lines.append(lis)

        return self.lines

# ========================计算new_m和old_m=================================
'''
dic的ABC
需要现在的质心点和现在的质心速度

'''

class CalculateM:
    def run_m(self , lis , centroid_x , centroid_y , centroid_vx , centroid_vy ):
        for i , params in enumerate(lis):
            A , B ,C = params
            post_x = centroid_x - centroid_vx
            post_y = centroid_y - centroid_vy
            m1 = A*post_x + B*post_y + C
            m2 = A*centroid_x + B*centroid_y + C
            result = m1*m2
            if result >= 0 :
                return False
            else:
                return True





if __name__ == '__main__':
    iou_tracker = IouTracker()
    iou_tracker.main()
