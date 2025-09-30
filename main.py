import sys
import os
import argparse
import time
import shutil
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QMessageBox, QSpinBox, QFormLayout
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

# --- FFmpeg Detection Logic ---

def check_ffmpeg():
    """
    Checks for FFmpeg and configures pydub.
    1. Checks for ffmpeg executable in the script's directory.
    2. Checks for ffmpeg in the system PATH.
    Returns True if found and configured, False otherwise.
    """
    ffmpeg_exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # Determine the script's directory. Handles bundled executables (PyInstaller).
    if getattr(sys, 'frozen', False):
        script_dir = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    local_ffmpeg_path = os.path.join(script_dir, ffmpeg_exe)

    # 1. Check for local ffmpeg executable (most reliable)
    if os.path.exists(local_ffmpeg_path):
        print(f"ローカルのFFmpegを検出しました: {local_ffmpeg_path}")
        AudioSegment.converter = local_ffmpeg_path
        return True

    # 2. Check for ffmpeg in system PATH
    if shutil.which(ffmpeg_exe):
        print("システムPATHでFFmpegを検出しました。")
        # pydub will find it automatically, no need to set AudioSegment.converter
        return True

    # 3. If all checks fail, FFmpeg is not available.
    return False

# --- Core Processing Logic (Shared between GUI and CUI) ---

def get_audio_files(directory):
    """指定されたディレクトリからWAVとMP3ファイルを取得する"""
    supported_formats = ('.wav', '.mp3')
    files = []
    if not os.path.isdir(directory):
        return None, f"エラー: 指定されたパスはディレクトリではありません: {directory}"

    for f in os.listdir(directory):
        if f.lower().endswith(supported_formats):
            files.append(os.path.join(directory, f))

    if not files:
        return None, f"警告: ディレクトリ内に処理対象の音声ファイルが見つかりません: {directory}"

    return files, None

# --- GPU/CPU Device Detection ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"処理デバイス: {DEVICE}")

def process_single_file(file_path, output_dir, output_format):
    """
    単一の音声ファイルをtorchaudioで処理する。GPUが利用可能ならGPUを使用する。
    成功した場合は出力パスを、失敗した場合はエラーメッセージを返す。
    """
    try:
        # 1. torchaudioで音声ファイルをロード
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.to(DEVICE)

        # 2. エフェクトチェーンを定義
        # compand: ダイナミックレンジ圧縮
        # loudness: 目標ラウドネス(-18 LUFS)にノーマライズ
        effects = [
            ["compand", "0.005,0.2", "6:-70,-60,-20", "-5", "-90", "0.2"],
            ["loudness", "-18.0"],
        ]

        # 3. エフェクトを適用
        processed_waveform, processed_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )

        # 4. ファイルをエクスポート
        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)

        if output_format == 'wav':
            output_path = os.path.join(output_dir, f"{file_name}.wav")
            torchaudio.save(output_path, processed_waveform.cpu(), processed_sample_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
        else:  # mp3
            output_path = os.path.join(output_dir, f"{file_name}.mp3")
            torchaudio.save(output_path, processed_waveform.cpu(), processed_sample_rate, format="mp3", bitrate=256)

        return output_path
    except Exception as e:
        return f"ファイル処理中にエラーが発生しました: {file_path}\n{e}"

def process_audio_files(file_list, output_dir, output_format, max_workers, progress_callback=None):
    """
    音声ファイルのリストを並列処理する。
    progress_callbackは進捗を報告するための関数(value, max_value)
    """
    os.makedirs(output_dir, exist_ok=True)
    errors = []

    # CUIモード用のプログレスバー
    progress_bar = None
    if progress_callback is None:
        progress_bar = tqdm(total=len(file_list), desc="オーディオファイルを処理中", unit="file")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, fp, output_dir, output_format): fp for fp in file_list}

        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if "エラー" in result:
                errors.append(result)

            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, len(file_list))
            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    if errors:
        # エラーメッセージを結合して返す
        return "\n".join(errors)

    return None # Success

# --- GUI Application ---

class ProcessingWorker(QObject):
    """バックグラウンドでファイル処理を行うワーカー"""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str) # エラーメッセージまたはNoneを渡す

    def __init__(self, files, output_dir, output_format, max_workers):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.output_format = output_format
        self.max_workers = max_workers

    def run(self):
        """処理を実行する"""
        error = process_audio_files(
            self.files,
            self.output_dir,
            self.output_format,
            self.max_workers,
            self.progress.emit
        )
        self.finished.emit(error)

class AudioNormalizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WAVs-Normalizer')
        self.setGeometry(100, 100, 450, 350) # Window size adjusted

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.input_dir = ''
        self.files_to_process = []
        self.start_time = 0
        self.processing_thread = None
        self.worker = None

        self.select_folder_button = QPushButton('入力フォルダを選択')
        self.select_folder_button.clicked.connect(self.select_input_folder)
        self.layout.addWidget(self.select_folder_button)

        self.status_label = QLabel('フォルダを選択してください。')
        self.layout.addWidget(self.status_label)

        # --- Options Layout ---
        form_layout = QFormLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV 16bit', 'MP3 256kbps'])
        form_layout.addRow('出力形式:', self.format_combo)

        self.worker_spinbox = QSpinBox()
        self.worker_spinbox.setMinimum(1)
        self.worker_spinbox.setMaximum(multiprocessing.cpu_count() * 2)
        self.worker_spinbox.setValue(multiprocessing.cpu_count())
        form_layout.addRow('並列処理数:', self.worker_spinbox)

        self.layout.addLayout(form_layout)
        # --- End Options Layout ---

        self.process_button = QPushButton('処理開始')
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

    def select_input_folder(self):
        self.input_dir = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if self.input_dir:
            files, err = get_audio_files(self.input_dir)
            if err:
                self.status_label.setText(err)
                self.process_button.setEnabled(False)
                return

            self.files_to_process = files
            file_count = len(self.files_to_process)
            self.status_label.setText(f'入力フォルダ: {self.input_dir}\n{file_count}個のファイルが見つかりました。')
            self.process_button.setEnabled(True)

    def start_processing(self):
        self.set_ui_enabled(False)
        output_dir = f"{self.input_dir}-Nomalized"
        output_format = 'wav' if 'WAV' in self.format_combo.currentText() else 'mp3'
        max_workers = self.worker_spinbox.value()

        total_files = len(self.files_to_process)
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"処理開始... (0/{total_files})")
        self.start_time = time.time()

        # Setup and start the background thread
        self.processing_thread = QThread()
        self.worker = ProcessingWorker(self.files_to_process, output_dir, output_format, max_workers)
        self.worker.moveToThread(self.processing_thread)

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.processing_thread.started.connect(self.worker.run)

        self.processing_thread.start()

    def on_processing_finished(self, error):
        self.processing_thread.quit()
        self.processing_thread.wait()

        total_files = len(self.files_to_process)
        if error:
            self.show_message("エラー", error, QMessageBox.Icon.Critical)
            self.status_label.setText("エラーが発生しました。")
        else:
            elapsed_time = time.time() - self.start_time
            self.status_label.setText(f"処理完了: {total_files}/{total_files} 件 (処理時間: {elapsed_time:.2f}秒)")
            self.show_message("完了", "処理が完了しました。", QMessageBox.Icon.Information)

        self.reset_ui()

    def update_progress(self, value, max_value):
        self.progress_bar.setValue(value)
        self.progress_bar.setMaximum(max_value)

        status_text = f"処理中... ({value}/{max_value})"

        if value > 0 and value < max_value:
            elapsed_time = time.time() - self.start_time
            avg_time_per_file = elapsed_time / value
            files_remaining = max_value - value
            estimated_remaining_time = avg_time_per_file * files_remaining
            mins, secs = divmod(estimated_remaining_time, 60)
            time_str = f"{int(mins):02d}:{int(secs):02d}"
            status_text += f" - 残り時間: 約{time_str}"

        self.status_label.setText(status_text)

    def set_ui_enabled(self, enabled):
        self.select_folder_button.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.worker_spinbox.setEnabled(enabled)
        self.process_button.setEnabled(enabled)

    def reset_ui(self):
        self.set_ui_enabled(True)
        self.status_label.setText('準備完了。次のフォルダを選択してください。')
        self.process_button.setEnabled(False)
        self.processing_thread = None
        self.worker = None

    def show_message(self, title, message, icon):
        msg_box = QMessageBox()
        msg_box.setIcon(icon)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.exec()

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Audio Normalizer for WAV and MP3 files.")
    parser.add_argument('--no-gui', action='store_true', help="Run in command-line interface mode.")
    parser.add_argument('-i', '--input', dest='input_dir', help="Input directory path (required for CUI).")
    parser.add_argument('-f', '--format', choices=['wav', 'mp3'], default='wav', help="Output format: wav or mp3 (for CUI).")
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of parallel worker threads (default: {multiprocessing.cpu_count()})."
    )
    args = parser.parse_args()

    # --- FFmpeg Dependency Check ---
    if not check_ffmpeg():
        msg = (
            "致命的なエラー: FFmpegが見つかりません。\n\n"
            "このアプリケーションを動作させるにはFFmpegが必要です。\n\n"
            "【解決策】\n"
            "1. (推奨) FFmpeg公式サイトから実行ファイルをダウンロードし、`ffmpeg.exe`をこのプログラム(`main.py`)と同じフォルダに置いてください。\n"
            "2. または、FFmpegをPCにインストールし、システムのPATH環境変数にその場所を登録してください。\n\n"
            "詳細はREADME.mdファイルをご確認ください。"
        )
        print(msg)
        if not args.no_gui:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "FFmpegが見つかりません", msg)
        sys.exit(1)

    if args.no_gui:
        if not args.input_dir:
            parser.error("--input is required for --no-gui mode.")

        print("CUIモードで処理を開始します。")
        print(f"入力フォルダ: {args.input_dir}")
        files, err = get_audio_files(args.input_dir)
        if err:
            print(err)
            sys.exit(1)

        print(f"{len(files)}個の音声ファイルを、{args.workers}個のワーカーで処理します。")
        output_dir = f"{args.input_dir}-Nomalized"

        start_time = time.time()
        error = process_audio_files(files, output_dir, args.format, args.workers)
        end_time = time.time()

        if error:
            print("\n処理がエラーにより中断されました。")
            print("--- エラー詳細 ---")
            print(error)
            print("--- エラー詳細 ---")
        else:
            print("\nすべてのファイルの処理が完了しました。")

        print(f"処理時間: {end_time - start_time:.2f}秒")
    else:
        app = QApplication(sys.argv)
        ex = AudioNormalizerGUI()
        ex.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    main()