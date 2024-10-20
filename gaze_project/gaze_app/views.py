from django.shortcuts import render
from django.http import HttpResponse, FileResponse
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from django.http import Http404
from django.conf import settings

# MediaPipeのFaceMeshを初期化
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def process_video_with_tracking(video_file):
    # 一時ファイルを作成し、アップロードされた動画を保存
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    for chunk in video_file.chunks():
        temp_file.write(chunk)
    temp_file.seek(0)

    # OpenCVで動画を読み込む
    cap = cv2.VideoCapture(temp_file.name)

    # 'mp4v'コーデックを指定して出力ファイルを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"tracked_{os.path.basename(temp_file.name)}"
    output_file_path = os.path.join(settings.MEDIA_ROOT, output_filename)

    # 動画の幅・高さ・フレームレートを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力用の動画ファイルを初期化
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    total_gaze_movement = 0  # 視線の総移動量を格納する変数

    # フレームごとに動画を解析
    while cap.isOpened():
        ret, frame = cap.read()  # フレームを1つ読み込む
        if not ret:
            break

        # BGRからRGBに変換（MediaPipeはRGBを要求）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipeで顔のランドマークを検出
        results = mp_face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 左目と右目のランドマークを取得
                left_eye_landmark = face_landmarks.landmark[145]
                right_eye_landmark = face_landmarks.landmark[374]

                # ランドマークの2D位置を画面上のピクセルに変換
                left_x, left_y = int(left_eye_landmark.x * width), int(left_eye_landmark.y * height)
                right_x, right_y = int(right_eye_landmark.x * width), int(right_eye_landmark.y * height)

                # 目のランドマークを可視化（緑の円）
                cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)  # 左目
                cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)  # 右目

                # 左目と右目間の距離を計算（2D平面でのユークリッド距離）
                gaze_movement = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)

                # 視線の総移動量を累積
                total_gaze_movement += gaze_movement

        # フレームにオーバーレイを追加して保存
        out.write(frame)

    cap.release()  # 動画のキャプチャを終了
    out.release()  # 出力動画の保存を終了
    temp_file.close()  # 一時ファイルを閉じる

    # 視線の総移動量と出力ファイルのパスを返す
    return total_gaze_movement, output_filename

def upload_video(request):
    analysis_result = None
    video_url = None

    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        if video_file.content_type == 'video/mp4':
            # 視線トラッキング付き動画の生成と解析結果の取得
            total_gaze_movement, tracked_video_path = process_video_with_tracking(video_file)

            # 動画のURLと解析結果を設定
            video_url = os.path.basename(tracked_video_path)
            analysis_result = f'Total Gaze Movement: {total_gaze_movement:.2f}'

            # 生成された動画をダウンロード可能にする
            request.session['tracked_video_path'] = tracked_video_path
            return render(request, 'upload.html', {
                'analysis_result': analysis_result,
                'video_url': video_url
            })
        else:
            return HttpResponse('Invalid file type. Please upload an MP4 video.')
    
    return render(request, 'upload.html', {'analysis_result': analysis_result, 'video_url': video_url})

from django.http import Http404

def download_video(request):
    # セッションから動画のパスを取得
    tracked_video_path = request.session.get('tracked_video_path')

    # 動画ファイルが存在するかを確認
    if tracked_video_path and os.path.exists(tracked_video_path):
        try:
            # 動画ファイルを開き直す
            video_file = open(tracked_video_path, 'rb')
            response = FileResponse(video_file, as_attachment=True, filename='tracked_video.mp4')
            return response
        except Exception as e:
            print(f"Error during file download: {e}")  # デバッグ用のログ
            raise Http404("Video file could not be downloaded.")
    else:
        # ファイルが見つからない場合
        raise Http404("No video available for download.")

def some_view(request):
    # 'upload.html'テンプレートをレンダリングする
    context = {
        'media_url': settings.MEDIA_URL  # MEDIA_URLをテンプレートに渡す
    }
    return render(request, 'upload.html', context)
