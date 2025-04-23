
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import torch
from PIL import Image, ImageTk
from threading import Thread

# 加载 YOLOv5 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
TARGET_CLASSES = ["person", "car"]

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能目标检测系统")
        self.root.geometry("800x600")
        self.root.resizable(True, True)  # 不禁止调整窗口大小

        # 视频相关参数
        self.cap = None
        self.is_running = False
        self.current_source = None

        # 创建界面组件
        self.create_widgets()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """创建 GUI 组件"""
        # 加载背景图片
        try:
            self.bg_image = Image.open("frc-2f8b8ae2e23ee5a59a508a95e3ba7ccd.png")  # 替换成你的实际背景图路径
            self.bg_image = self.bg_image.resize((800, 600), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)

            # 创建 Canvas 并设置背景
            self.canvas = tk.Canvas(self.root, width=800, height=600)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)
            self.canvas.place(x=0, y=0, relwidth=1, relheight=1)  # 让 Canvas 作为背景
        except Exception as e:
            print(f"加载背景图失败: {e}")

        # 控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制面板", padding=(10, 5))
        control_frame.place(relx=0.5, rely=0.05, anchor=tk.N)

        # 按钮
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12, 'bold'), padding=10)
        
        btn_open_video = ttk.Button(control_frame, text="打开本地视频", command=self.open_local_video, style='TButton')
        btn_open_camera = ttk.Button(control_frame, text="打开摄像头", command=self.open_camera, style='TButton')
        btn_stop = ttk.Button(control_frame, text="停止检测", command=self.stop_detection, style='TButton')

        btn_open_video.pack(side=tk.LEFT, padx=5)
        btn_open_camera.pack(side=tk.LEFT, padx=5)
        btn_stop.pack(side=tk.RIGHT, padx=5)

        # 视频显示区域（放在最上层）
        self.video_frame = ttk.Label(self.root, borderwidth=3, relief="ridge")
        self.video_frame.place(relx=0.5, rely=0.55, anchor=tk.CENTER, width=640, height=480)

    def open_local_video(self):
        """打开本地视频"""
        path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4;*.avi;*.mov")])
        if path:
            self.start_detection(path)

    def open_camera(self):
        """打开摄像头"""
        self.start_detection(0)

    def start_detection(self, source):
        """启动检测线程"""
        if self.is_running:
            self.stop_detection()

        self.current_source = source
        self.is_running = True
        Thread(target=self.detection_loop, daemon=True).start()

    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def detection_loop(self):
        """检测主循环"""
        self.cap = cv2.VideoCapture(self.current_source)
        if not self.cap.isOpened():
            return

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLOv5 检测
            results = model(frame)
            detections = results.pred[0].cpu().numpy()

            # 绘制检测结果
            processed_frame = self.draw_boxes(frame, detections)

            # 显示处理后的帧
            self.update_display(processed_frame)

        self.cap.release()
        self.cap = None

    def draw_boxes(self, frame, detections):
        """绘制检测框"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for *xyxy, conf, cls in detections:
            label = model.names[int(cls)]
            if label in TARGET_CLASSES:
                color = (0, 255, 0) if label == "person" else (0, 0, 255)
                cv2.rectangle(frame, 
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (int(xyxy[0]), int(xyxy[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    #def update_display(self, frame):
        """更新视频显示"""
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

    def update_display(self, frame):
        """更新视频显示"""
    # 获取视频显示区域的宽度和高度
        display_width = self.video_frame.winfo_width()
        display_height = self.video_frame.winfo_height()

    # 获取原始帧的宽度和高度
        frame_height, frame_width, _ = frame.shape

    # 计算缩放比例
        aspect_ratio = frame_width / frame_height
        if display_width / display_height > aspect_ratio:
        # 如果显示区域的宽高比大于视频帧的宽高比，按高度缩放
            new_height = display_height
            new_width = int(new_height * aspect_ratio)
        else:
        # 如果显示区域的宽高比小于或等于视频帧的宽高比，按宽度缩放
            new_width = display_width
            new_height = int(new_width / aspect_ratio)

    # 缩放视频帧
        resized_frame = cv2.resize(frame, (new_width, new_height))

    # 将缩放后的帧转换为 Image 对象
        img = Image.fromarray(resized_frame)
        imgtk = ImageTk.PhotoImage(image=img)

    # 更新 Canvas 显示
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)


    def on_close(self):
        """关闭窗口时的清理操作"""
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
