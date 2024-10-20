from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from django.conf import settings

# MediaPipeのFaceMeshを初期化
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from django.conf import settings

# MediaPipeのFaceMeshを初期化
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def save_temp_video(video_file):
    """動画ファイルを一時保存する"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    for chunk in video_file.chunks():
        temp_file.write(chunk)
    temp_file.seek(0)
    return temp_file

def initialize_video_writer(cap, output_filename):
    """動画出力用の初期化を行う"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file_path = os.path.join(settings.MEDIA_ROOT, output_filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
    return out, width, height

def calculate_gaze_movement(center_x, center_y, prev_center_x, prev_center_y):
    """視線の移動量を計算（絶対値を使用）"""
    dx = abs(center_x - prev_center_x)
    dy = abs(center_y - prev_center_y)
    return dx + dy

def process_frame(frame, prev_center_x, prev_center_y, width, height):
    """フレームごとの視線の処理を行う"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(frame_rgb)
    center_x, center_y = None, None
    gaze_movement = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmark = face_landmarks.landmark[145]
            right_eye_landmark = face_landmarks.landmark[374]

            # 両目の中心を計算
            center_x = (left_eye_landmark.x + right_eye_landmark.x) / 2 * width
            center_y = (left_eye_landmark.y + right_eye_landmark.y) / 2 * height

            # 視線の移動量を計算
            if prev_center_x is not None and prev_center_y is not None:
                gaze_movement = calculate_gaze_movement(center_x, center_y, prev_center_x, prev_center_y)

            # 中心位置を可視化
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

    return frame, center_x, center_y, gaze_movement

def process_video_with_tracking(video_file):
    """動画内の視線トラッキングを行い、平均視線移動量を計算"""
    temp_file = save_temp_video(video_file)
    cap = cv2.VideoCapture(temp_file.name)
    
    output_filename = f"tracked_{os.path.basename(temp_file.name)}"
    out, width, height = initialize_video_writer(cap, output_filename)

    total_gaze_movement = 0
    prev_center_x, prev_center_y = None, None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, center_x, center_y, gaze_movement = process_frame(
            frame, prev_center_x, prev_center_y, width, height
        )

        total_gaze_movement += gaze_movement
        prev_center_x, prev_center_y = center_x, center_y

        frame_count += 1
        out.write(frame)

    cap.release()
    out.release()
    temp_file.close()

    avg_gaze_movement = total_gaze_movement / frame_count if frame_count > 0 else 0
    return avg_gaze_movement, output_filename


def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        if video_file.content_type == 'video/mp4':
            avg_gaze_movement, tracked_video_path = process_video_with_tracking(video_file)

            video_url = os.path.basename(tracked_video_path)
            analysis_result = f'Average Gaze Movement: {avg_gaze_movement:.2f}'

            return JsonResponse({
                'analysis_result': analysis_result,
                'video_url': video_url
            })
        else:
            return JsonResponse({'error': 'Invalid file type. Please upload an MP4 video.'}, status=400)

    return render(request, 'upload.html')

def download_video(request):
    tracked_video_path = request.session.get('tracked_video_path')

    if tracked_video_path and os.path.exists(tracked_video_path):
        try:
            video_file = open(tracked_video_path, 'rb')
            response = FileResponse(video_file, as_attachment=True, filename='tracked_video.mp4')
            return response
        except Exception as e:
            print(f"Error during file download: {e}")
            raise Http404("Video file could not be downloaded.")
    else:
        raise Http404("No video available for download.")

def some_view(request):
    context = {
        'media_url': settings.MEDIA_URL
    }
    return render(request, 'upload.html', context)
