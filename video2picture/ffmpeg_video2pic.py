import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=50):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 如果当前帧是需要保存的帧
        if frame_count % frame_interval == 0:
            # 保存帧为图片
            image_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(image_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames extracted and saved in {output_folder}")


def process_videos(input_folder):
    # 遍历文件夹中的所有视频文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.mkv', '.avi')):  # 支持的文件格式
            video_path = os.path.join(input_folder, filename)
            output_folder = os.path.join(input_folder, os.path.splitext(filename)[0])
            extract_frames(video_path, output_folder)


# 输入视频文件夹路径
input_folder = 'D:/dataset/'

process_videos(input_folder)