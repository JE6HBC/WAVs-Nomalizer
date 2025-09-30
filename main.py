import sys
import os
import argparse
import time
from tqdm import tqdm
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QMessageBox
)
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

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

            # 会話向けのコンプレッサー設定
            compressed_audio = compress_dynamic_range(
                audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0
            )
            normalized_audio = compressed_audio.normalize()

            # ファイルの書き出し
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
            return error_message # Stop processing on error

        if progress_callback:
            # Report progress after completing a file (i+1)
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

        # --- UI Elements ---
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

        # This is a simplified progress update. For non-blocking UI, threading is needed.
        # For this MVP, we'll accept that the UI might freeze during processing.
        error = process_audio_files(self.files_to_process, output_dir, output_format, self.update_progress)

        if error:
            self.show_message("エラー", error, QMessageBox.Icon.Critical)
        else:
            # Final update to show completion
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

            # Format remaining time
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
        # Don't reset progress bar value immediately, so user can see it's 100%
        # self.progress_bar.setValue(0)
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
    try:
        # pydub checks for ffmpeg presence when trying to use it.
        # We can try to load a null file to trigger this check.
        AudioSegment.from_file(os.devnull)
    except Exception:
        msg = "致命的なエラー: FFmpegが見つかりません。README.mdの指示に従ってインストールし、PATHを設定してください。"
        print(msg)
        if not args.no_gui:
            # Dummy app to show message box
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "エラー", msg)
        sys.exit(1)


    if args.no_gui:
        # --- CUI Mode ---
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
        # --- GUI Mode ---
        app = QApplication(sys.argv)
        ex = AudioNormalizerGUI()
        ex.show()
        sys.exit(app.exec())


if __name__ == '__main__':
    main()