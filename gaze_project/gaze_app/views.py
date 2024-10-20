from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# MediaPipeのFaceMeshを初期化
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,   # 動画処理モード（静止画像ではなく動画モード）
    max_num_faces=1,           # 処理する最大の顔数（ここでは1つの顔のみ）
    refine_landmarks=True,     # ランドマークの精度を向上
    min_detection_confidence=0.5  # 検出の最小信頼度
)

def process_video(video_file):
    # 一時ファイルを作成し、アップロードされた動画を保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    for chunk in video_file.chunks():
        temp_file.write(chunk)
    temp_file.seek(0)  # ファイルの先頭に戻す

    # OpenCVで動画を読み込む
    cap = cv2.VideoCapture(temp_file.name)
    total_gaze_movement = 0  # 視線の総移動量を計算する変数

    # フレームごとに動画を解析
    while cap.isOpened():
        ret, frame = cap.read()  # フレームを1つ読み込む
        if not ret:
            break  # フレームが読み込めない場合、ループを終了

        # BGRからRGBに変換（MediaPipeはRGBを必要とするため）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipeを使って顔のランドマークを検出
        results = mp_face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 左目と右目のランドマークを取得
                left_eye_landmark = face_landmarks.landmark[145]   # 左目の特定のランドマーク
                right_eye_landmark = face_landmarks.landmark[374]  # 右目の特定のランドマーク

                # 左右の目のランドマーク間の距離を計算（2Dの距離）
                gaze_movement = np.sqrt(
                    (right_eye_landmark.x - left_eye_landmark.x) ** 2 +
                    (right_eye_landmark.y - left_eye_landmark.y) ** 2
                )
                # 視線移動量の総量を累積
                total_gaze_movement += gaze_movement

    cap.release()  # 動画のキャプチャを終了
    temp_file.close()  # 一時ファイルを閉じる

    # 視線の総移動量を結果として返す
    return f'Total Gaze Movement: {total_gaze_movement:.2f}'

def upload_video(request):
    analysis_result = None  # 解析結果を保持する変数
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        if video_file.content_type == 'video/mp4':
            # アップロードされた動画ファイルを解析
            analysis_result = process_video(video_file)

            # 解析結果をテンプレートに渡してレンダリング
            return render(request, 'upload.html', {'analysis_result': analysis_result})
        else:
            return HttpResponse('Invalid file type. Please upload an MP4 video.')
    
    # 初期のページロード時は解析結果なしでページを表示
    return render(request, 'upload.html', {'analysis_result': analysis_result})
