# Django-MediaPipe-demo
This is a minimal demo of using MediaPipe in a web app built with Django to quantify the eye movement of a person in a video file sent to it.

# ローカル環境でのセットアップ手順

システム依存パッケージのインストール (Ubuntuの場合)
以下のコマンドを実行して、必要なシステム依存パッケージをインストールしてください。

bash
コードをコピーする
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 libglib2.0-0 libxrender1
Python仮想環境のセットアップ
仮想環境を作成して有効化します。
デプロイする場合はDocker イメージを作成

bash
コードをコピーする
python3 -m venv venv
source venv/bin/activate
依存パッケージをインストールします。

bash
コードをコピーする
pip install -r requirements.txt
Djangoアプリの起動
データベースのマイグレーションを適用します。

bash
コードをコピーする
python manage.py migrate
開発サーバーを起動します。

bash
コードをコピーする
python manage.py runserver

# 要件定義書（MVP - 最小構成、保存・認証・ホスティングなし）

## 1. 機能要件
1. 動画アップロード機能
   - ユーザーがローカルデバイスから動画ファイル（MP4形式）をアップロードし、その場でメモリ上に保持して解析を行うシンプルなインターフェースを提供。

2. 視線検出と解析機能
   - アップロードされた動画をフレーム単位で処理し、顔と目のランドマークを検出して視線移動量を計算。
   - MediaPipeまたはOpenCVとDlibを使って目のランドマークを取得し、フレーム間の視線位置変化を計算する。
   - 動画データは解析中のみメモリに保持し、解析が終了次第即座に破棄。

3. 解析結果の表示
   - シンプルな結果ページで、視線移動の数値評価（例: 移動距離の総量）をテキストとして表示。
   - 視線の移動量や安定性スコアを簡単に確認できる。

## 2. 非機能要件
1. シンプルなUI
   - 動画をアップロードし、解析結果をすぐに表示できるように、最小限のHTMLフォームを使用。
   - JavaScriptを用いた進行状況表示（例: ロード中スピナーなど）を実装。

2. 動画のメモリ処理
   - 動画はアップロード時にメモリ上に保持し、サーバーに保存せずにそのまま解析を行う。
   - 処理が完了したら動画データは破棄され、サーバーに残らない。

3. ローカル動作のみ
   - アプリケーションはローカルデバイス上で動作し、ホスティングやクラウドサービスは不要。
   - ローカル開発サーバー（Djangoの開発用サーバー）を利用し、ローカルでのMVP検証が可能。

## 3. 使用技術
- バックエンド: Django（最小限の構成）
- フロントエンド: シンプルなHTMLとJavaScript
- 視線解析ライブラリ: MediaPipe（顔と目のランドマーク検出を使用）
- ストレージ: なし（メモリ上でのみ処理）

## 4. ディレクトリ構成

```plaintext
project_root/
│
├── gaze_app/              # アプリケーションのメインディレクトリ
│   ├── static/            # 静的ファイル（CSS, JavaScriptなど）
│   ├── templates/         # HTMLテンプレート
│   │   ├── upload.html    # 動画アップロード用のページ
│   │   └── result.html    # 解析結果を表示するページ
│   ├── __init__.py
│   ├── apps.py
│   ├── views.py           # ビュー（アップロードと解析処理を定義）
│   ├── urls.py            # アプリケーションのルーティング設定
│   └── models.py          # モデル定義（MVPでは不要、空のまま）
│
├── manage.py              # Djangoの管理コマンド
├── requirements.txt       # 必要なパッケージのリスト（Django, MediaPipeなど）
├── templates/             # 共通のテンプレートディレクトリ
└── .gitignore             # Gitで追跡しないファイルのリスト
```

### 備考
- 動画保存なし: 動画はアップロード時にメモリ上でのみ保持し、保存しません。
- 認証機能なし: ローカル環境での検証のため、ユーザー認証は不要。
- ホスティング不要: ローカルデバイス上で開発用サーバーを使い、動作検証を行います。
