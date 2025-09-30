import sys
import os
import argparse
import time
import shutil
from tqdm import tqdm
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QMessageBox
)
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

def process_audio_files(file_list, output_dir, output_format, progress_callback=None):
    """
    音声ファイルのリストを処理する。
    progress_callbackは進捗を報告するための関数(value, max_value)
    """
    os.makedirs(output_dir, exist_ok=True)

    iterator = tqdm(file_list, desc="Processing audio files", unit="file") if progress_callback is None else file_list

    for i, file_path in enumerate(iterator):
        try:
            audio = AudioSegment.from_file(file_path)

            compressed_audio = compress_dynamic_range(
                audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0
            )
            normalized_audio = compressed_audio.normalize()

            base_name = os.path.basename(file_path)
            if output_format == 'wav':
                output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.wav")
                normalized_audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
            else: # mp3
                output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.mp3")
                normalized_audio.export(output_path, format="mp3", bitrate="256k")
        except Exception as e:
            error_message = f"ファイル処理中にエラーが発生しました: {file_path}\n{e}"
            if progress_callback is None: # CUI mode
                print(f"\n{error_message}")
            return error_message

        if progress_callback:
            progress_callback(i + 1, len(file_list))

    return None # Success

# --- GUI Application ---

class AudioNormalizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WAVs-Normalizer')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.input_dir = ''
        self.files_to_process = []
        self.start_time = 0

        self.select_folder_button = QPushButton('入力フォルダを選択')
        self.select_folder_button.clicked.connect(self.select_input_folder)
        self.layout.addWidget(self.select_folder_button)

        self.status_label = QLabel('フォルダを選択してください。')
        self.layout.addWidget(self.status_label)

        self.format_label = QLabel('出力形式:')
        self.layout.addWidget(self.format_label)

        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV 16bit', 'MP3 256kbps'])
        self.layout.addWidget(self.format_combo)

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
        output_format_text = self.format_combo.currentText()
        output_format = 'wav' if 'WAV' in output_format_text else 'mp3'

        total_files = len(self.files_to_process)
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"処理中... (0/{total_files})")

        self.start_time = time.time()

        error = process_audio_files(self.files_to_process, output_dir, output_format, self.update_progress)

        if error:
            self.show_message("エラー", error, QMessageBox.Icon.Critical)
        else:
            self.status_label.setText(f"処理完了: {total_files}/{total_files} 件")
            self.show_message("完了", "処理が完了しました。", QMessageBox.Icon.Information)

        self.reset_ui()

    def update_progress(self, value, max_value):
        self.progress_bar.setValue(value)
        self.progress_bar.setMaximum(max_value)
        elapsed_time = time.time() - self.start_time
        status_text = f"処理中... ({value}/{max_value})"

        if value > 0 and value < max_value:
            avg_time_per_file = elapsed_time / value
            files_remaining = max_value - value
            estimated_remaining_time = avg_time_per_file * files_remaining
            mins, secs = divmod(estimated_remaining_time, 60)
            time_str = f"{int(mins):02d}:{int(secs):02d}"
            status_text += f" - 残り時間: 約{time_str}"

        self.status_label.setText(status_text)
        QApplication.processEvents()

    def set_ui_enabled(self, enabled):
        self.select_folder_button.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.process_button.setEnabled(enabled)

    def reset_ui(self):
        self.set_ui_enabled(True)
        self.status_label.setText('準備完了。次のフォルダを選択してください。')
        self.process_button.setEnabled(False)

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

        print(f"入力フォルダ: {args.input_dir}")
        files, err = get_audio_files(args.input_dir)
        if err:
            print(err)
            sys.exit(1)

        print(f"{len(files)}個の音声ファイルを処理します。")
        output_dir = f"{args.input_dir}-Nomalized"
        error = process_audio_files(files, output_dir, args.format)

        if error:
            print("\n処理がエラーにより中断されました。")
        else:
            print("\nすべてのファイルの処理が完了しました。")
    else:
        app = QApplication(sys.argv)
        ex = AudioNormalizerGUI()
        ex.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    main()