import sys
import os
import asyncio
import json
import tempfile
import base64
import re
from datetime import datetime
from io import BytesIO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QComboBox, QLineEdit, 
                            QTabWidget, QScrollArea, QSplitter, QFileDialog, QMessageBox,
                            QDialog, QFormLayout, QSpinBox, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QBuffer, QPoint, QRect, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPainterPath, QBrush, QPen, QAction, QKeySequence, QShortcut, QPalette, QLinearGradient
from PyQt6 import QtCore
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import pyperclip
import requests
from PIL import ImageGrab, Image
import edge_tts
from groq import Groq
from openai import OpenAI
from queue import Queue

# Default settings
DEFAULT_SETTINGS = {
    "api_keys": {
        "groq": "",
        "openai_compatible": ""
    },
    "base_urls": {
        "openai_compatible": ""
    },
    "models": {
        "whisper": "whisper-large-v3-turbo",
        "llm": "gemini-2.0-flash"
    },
    "tts_voice": "en-US-AriaNeural",
    "tts_speed": "+0%",  # New setting for TTS speed
    "tts_volume": 1.0,   # New setting for TTS volume (0.0-1.0)
    "hotkeys": {
        "start_recording": "ctrl+[",
        "stop_recording": "ctrl+]",
        "take_screenshot": "ctrl+p",
        "delete_last_screenshot": "ctrl+\\",
        "stop_tts": "ctrl+'"
    },
    "microphone_index": 0,
    "max_conversation_history": 10,
    "system_prompt": "You are a helpful assistant with both voice and vision capabilities. Please provide concise and accurate responses."
}

class BlurredWidget(QWidget):
    """Custom widget with a blurred background effect"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.blur_opacity = 230  # 0-255
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create blur effect using a semi-transparent background
        color = QColor(240, 240, 240, self.blur_opacity)  # Light color with transparency
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 15, 15)
        painter.fillPath(path, color)
        
        # Add subtle gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 30))
        gradient.setColorAt(1, QColor(0, 0, 0, 15))
        painter.fillPath(path, QBrush(gradient))
        
        # Add border
        pen = QPen(QColor(200, 200, 200, 100), 1)
        painter.setPen(pen)
        painter.drawRoundedRect(0, 0, self.width()-1, self.height()-1, 15, 15)

class AudioRecorder(QThread):
    """Thread for recording audio"""
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Common for speech recognition
        
    def run(self):
        self.is_recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.is_recording:
                self.audio_data.append(indata.copy())
                
        try:
            mic_index = self.settings.get("microphone_index", 0)
            with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate,
                               device=mic_index if mic_index >= 0 else None):
                self.status_update.emit("Recording started...")
                while self.is_recording:
                    sd.sleep(100)
                    
            if len(self.audio_data) > 0:
                audio = np.concatenate(self.audio_data, axis=0)
                temp_file = tempfile.mktemp(suffix=".wav")
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 bytes = 16 bits
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio * 32767).astype(np.int16).tobytes())
                self.status_update.emit("Recording saved")
                self.finished.emit(temp_file)
            else:
                self.status_update.emit("No audio recorded")
                self.finished.emit("")
                
        except Exception as e:
            self.status_update.emit(f"Recording error: {str(e)}")
            self.finished.emit("")
    
    def stop(self):
        self.is_recording = False

class TranscriptionThread(QThread):
    """Thread for transcribing audio using Groq's Whisper API"""
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, audio_file, settings):
        super().__init__()
        self.audio_file = audio_file
        self.settings = settings
        
    def run(self):
        try:
            self.status_update.emit("Transcribing audio...")
            client = Groq(api_key=self.settings["api_keys"]["groq"])
            
            with open(self.audio_file, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(self.audio_file, file.read()),
                    model=self.settings["models"]["whisper"],
                    response_format="text"
                )
                
            self.status_update.emit("Transcription complete")
            
            # Handle the response based on its type
            if isinstance(transcription, str):
                # The API returned a string directly
                self.finished.emit(transcription)
            elif hasattr(transcription, 'text'):
                # The API returned an object with a text attribute
                self.finished.emit(transcription.text)
            else:
                # Try to convert to string if it's neither of the above
                self.finished.emit(str(transcription))
            
        except Exception as e:
            self.status_update.emit(f"Transcription error: {str(e)}")
            self.finished.emit("")

class LLMThread(QThread):
    """Thread for interacting with the language model"""
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)
    token_received = pyqtSignal(str)
    sentence_complete = pyqtSignal(str)
    
    def __init__(self, messages, settings, image_paths=None):
        super().__init__()
        self.messages = messages
        self.settings = settings
        self.image_paths = image_paths or []
        self.current_sentence = ""
        # Support both English and Chinese punctuation
        self.sentence_end_pattern = re.compile(r'[.!?。！？；;](\s|$)')
        
    def run(self):
        try:
            self.status_update.emit("Sending to AI...")
            
            # Prepare the client based on the available API key and base URL
            if self.settings["api_keys"]["openai_compatible"]:
                client = OpenAI(
                    api_key=self.settings["api_keys"]["openai_compatible"],
                    base_url=self.settings["base_urls"]["openai_compatible"]
                )
            else:
                self.status_update.emit("No valid API key found")
                self.finished.emit("")
                return
            
            # Prepare messages with images if any
            messages_with_images = self.messages.copy()
            
            # If there are images, add them to the last user message
            if self.image_paths and len(self.messages) > 0:
                # Find the last user message
                for i in range(len(messages_with_images) - 1, -1, -1):
                    if messages_with_images[i]["role"] == "user":
                        last_user_msg_idx = i
                        break
                else:
                    # If no user message found, add a new one
                    messages_with_images.append({"role": "user", "content": []})
                    last_user_msg_idx = len(messages_with_images) - 1
                
                # Convert the last user message to the content list format if it's a string
                if isinstance(messages_with_images[last_user_msg_idx]["content"], str):
                    text_content = messages_with_images[last_user_msg_idx]["content"]
                    messages_with_images[last_user_msg_idx]["content"] = [
                        {"type": "text", "text": text_content}
                    ]
                
                # Add each image to the content
                for img_path in self.image_paths:
                    try:
                        with open(img_path, "rb") as img_file:
                            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                            messages_with_images[last_user_msg_idx]["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
                    except Exception as e:
                        self.status_update.emit(f"Error processing image {img_path}: {str(e)}")
            
            # Stream the response
            full_response = ""
            self.current_sentence = ""
            response = client.chat.completions.create(
                model=self.settings["models"]["llm"],
                messages=messages_with_images,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    self.current_sentence += content
                    self.token_received.emit(content)
                    
                    # Check if we have a complete sentence
                    if self.sentence_end_pattern.search(content):
                        # We found a sentence end, emit the sentence
                        sentence = self.current_sentence.strip()
                        if sentence:
                            self.sentence_complete.emit(sentence)
                        self.current_sentence = ""
            
            # Emit any remaining content as a sentence
            if self.current_sentence.strip():
                self.sentence_complete.emit(self.current_sentence.strip())
            
            self.status_update.emit("AI response complete")
            self.finished.emit(full_response)
            
        except Exception as e:
            self.status_update.emit(f"AI error: {str(e)}")
            self.finished.emit("")

class TTSAudioConverter(QThread):
    """专门用于文本到语音音频文件转换的线程，不负责播放"""
    finished = pyqtSignal(str)  # 发送生成的音频文件路径
    status_update = pyqtSignal(str)
    
    def __init__(self, text, settings):
        super().__init__()
        # 清理文本中不应该被朗读的符号
        self.text = self.clean_text_for_speech(text)
        self.settings = settings
        self.stop_requested = False
        self.loop = None
    
    def clean_text_for_speech(self, text):
        """Clean text by removing symbols that shouldn't be spoken"""
        # Remove or replace special symbols
        cleaned_text = text
        
        # Remove asterisks, underscores (commonly used for emphasis in text)
        cleaned_text = re.sub(r'\*+', ' ', cleaned_text)  # Replace * with space
        cleaned_text = re.sub(r'_+', ' ', cleaned_text)   # Replace _ with space
        
        # Remove code formatting symbols
        cleaned_text = re.sub(r'`+', '', cleaned_text)    # Remove backticks
        
        # Remove URL brackets
        cleaned_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned_text)  # Replace [text](URL) with just text
        
        # Remove other special characters that don't need to be spoken
        cleaned_text = re.sub(r'[\#\~\^\{\}\|\<\>]', '', cleaned_text)
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    def run(self):
        try:
            self.status_update.emit(f"正在转换音频: '{self.text[:20]}...'")
            
            # 为音频创建临时文件
            temp_file = tempfile.mktemp(suffix=".mp3")
            
            # 在此线程中创建并设置事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # 使用Edge TTS将文本转换为语音，并应用指定的速率和音量
            async def convert_to_speech():
                voice = self.settings["tts_voice"]
                rate = self.settings.get("tts_speed", "+0%")
                
                # 根据Edge TTS文档格式化音量
                # Edge TTS需要音量为类似"+0%"（默认），"+50%"（更大声）或"-50%"（更小声）的字符串
                volume_float = self.settings.get("tts_volume", 1.0)
                
                # 将0.0-1.0转换为-100%到+100%之间的百分比
                # 0.0 -> -100%, 0.5 -> 0%, 1.0 -> +100%
                volume_percent = int((volume_float * 2 - 1) * 100)
                # 限制到合理范围（-100%到+100%）
                volume_percent = max(-100, min(100))
                # 按照Edge TTS要求格式化
                volume = f"{volume_percent:+d}%"
                
                # 创建带有语音、速率和音量选项的通信对象
                communicate = edge_tts.Communicate(self.text, voice, rate=rate, volume=volume)
                await communicate.save(temp_file)
            
            # 在此线程的事件循环中运行异步函数
            self.loop.run_until_complete(convert_to_speech())
            
            if self.stop_requested:
                self.status_update.emit("TTS转换已取消")
                try:
                    os.remove(temp_file)
                except:
                    pass
                # 正确关闭循环
                self.loop.close()
                self.finished.emit("")
                return
                
            # 正确关闭循环以防止"Event loop is closed"错误
            self.loop.close()
            
            self.status_update.emit("TTS转换完成")
            self.finished.emit(temp_file)  # 发送生成的音频文件路径
            
        except Exception as e:
            self.status_update.emit(f"TTS转换错误: {str(e)}")
            # 确保即使出错也关闭循环
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            self.finished.emit("")
    
    def stop(self):
        self.stop_requested = True
        # 确保如果存在事件循环则正确关闭它
        if self.loop and not self.loop.is_closed():
            async def close_loop():
                pass
            try:
                self.loop.run_until_complete(close_loop())
                self.loop.close()
            except:
                pass


class AudioPlayer(QThread):
    """专门用于播放音频文件的线程"""
    finished = pyqtSignal()
    status_update = pyqtSignal(str)
    
    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file
        self.stop_requested = False
        
    def run(self):
        try:
            if not os.path.exists(self.audio_file):
                self.status_update.emit("找不到音频文件")
                self.finished.emit()
                return
                
            self.status_update.emit("播放音频...")
            
            # 播放音频
            data, samplerate = sf.read(self.audio_file)
            sd.play(data, samplerate)
            sd.wait()
            
            if self.stop_requested:
                self.status_update.emit("音频播放已停止")
                self.finished.emit()
                return
            
            self.status_update.emit("音频播放完成")
            
            # 删除临时文件
            try:
                os.remove(self.audio_file)
            except:
                pass
                
            self.finished.emit()
            
        except Exception as e:
            self.status_update.emit(f"音频播放错误: {str(e)}")
            self.finished.emit()
    
    def stop(self):
        self.stop_requested = True
        sd.stop()


class ParallelTTSProcessor:
    """管理TTS文本并行处理和顺序播放的类"""
    def __init__(self, settings):
        self.settings = settings
        self.tts_queue = Queue()  # 文本队列
        self.audio_queue = Queue()  # 已转换音频文件队列
        
        self.is_processing = False
        self.is_playing = False
        
        # 当前活动线程
        self.current_converter = None
        self.current_player = None
        self.next_converter = None  # 用于并行准备下一个音频
        
        # 状态回调
        self.status_callback = None
        
        # 调试标志
        self.debug = True
        
    def set_status_callback(self, callback):
        """设置状态更新的回调函数"""
        self.status_callback = callback
        
    def update_status(self, message):
        """更新状态消息"""
        if self.status_callback:
            self.status_callback(message)
        elif self.debug:
            print(f"TTS状态: {message}")
        
    def add_text(self, text):
        """添加文本到TTS队列"""
        if text and text.strip():
            self.update_status(f"添加文本到队列: '{text[:20]}...'")
            self.tts_queue.put(text)
            
            # 如果处理器尚未启动，则开始处理
            if not self.is_processing:
                self.update_status("启动TTS处理")
                self.start_processing()
                
    def start_processing(self):
        """开始处理TTS队列"""
        self.is_processing = True
        self.process_next_text()
        
    def process_next_text(self):
        """处理队列中的下一个文本"""
        if self.tts_queue.empty():
            if not self.is_playing and self.audio_queue.empty():
                self.is_processing = False
                self.update_status("TTS处理完成")
            return
        
        # 获取下一个文本
        text = self.tts_queue.get()
        self.update_status(f"开始转换文本: '{text[:20]}...'")
        
        # 创建并启动转换器
        self.current_converter = TTSAudioConverter(text, self.settings)
        self.current_converter.status_update.connect(self.update_status)
        self.current_converter.finished.connect(self.on_conversion_finished)
        self.current_converter.start()
        
        # 如果队列中还有更多文本，立即开始并行转换下一个
        if not self.tts_queue.empty() and self.next_converter is None:
            next_text = self.tts_queue.get()
            self.update_status(f"开始并行转换下一段文本: '{next_text[:20]}...'")
            self.next_converter = TTSAudioConverter(next_text, self.settings)
            self.next_converter.status_update.connect(self.update_status)
            self.next_converter.finished.connect(self.on_next_conversion_finished)
            self.next_converter.start()
    
    def on_conversion_finished(self, audio_file):
        """当前文本转换完成的回调"""
        if audio_file:
            self.update_status("当前转换完成，添加到音频队列")
            # 将转换好的音频加入队列
            self.audio_queue.put(audio_file)
            # 如果没有正在播放，开始播放
            if not self.is_playing:
                self.update_status("开始播放第一段音频")
                self.play_next_audio()
        else:
            self.update_status("当前转换失败或被取消")
        
        # 清除完成的转换器引用
        temp_converter = self.current_converter
        self.current_converter = None
        
        # 确保线程对象能够被垃圾回收
        if temp_converter:
            temp_converter.deleteLater()
                
        # 处理队列中的下一个文本，但仅当没有正在并行转换的任务时
        if not self.next_converter:
            self.update_status("检查是否有更多文本需要处理")
            self.process_next_text()
    
    def on_next_conversion_finished(self, audio_file):
        """并行转换的下一个文本完成的回调"""
        if audio_file:
            self.update_status("并行转换完成，添加到音频队列")
            # 将转换好的音频加入队列
            self.audio_queue.put(audio_file)
            # 如果没有正在播放，开始播放
            if not self.is_playing:
                self.update_status("开始播放并行转换的音频")
                self.play_next_audio()
        else:
            self.update_status("并行转换失败或被取消")
        
        # 清除完成的转换器引用
        temp_converter = self.next_converter
        self.next_converter = None
        
        # 确保线程对象能够被垃圾回收
        if temp_converter:
            temp_converter.deleteLater()
        
        # 只有当当前转换也完成时，才处理下一个
        if not self.current_converter:
            self.update_status("尝试处理更多文本")
            self.process_next_text()
            
    def play_next_audio(self):
        """播放队列中的下一个音频"""
        if self.audio_queue.empty():
            self.is_playing = False
            self.update_status("音频队列为空")
            # 如果没有更多的文本要处理，处理完成
            if self.tts_queue.empty() and not self.current_converter and not self.next_converter:
                self.is_processing = False
                self.update_status("所有TTS处理和播放完成")
            return
        
        self.is_playing = True
        audio_file = self.audio_queue.get()
        
        self.update_status(f"开始播放音频: {os.path.basename(audio_file)}")
        
        # 创建并启动播放器
        self.current_player = AudioPlayer(audio_file)
        self.current_player.status_update.connect(self.update_status)
        self.current_player.finished.connect(self.on_audio_finished)
        self.current_player.start()
    
    def on_audio_finished(self):
        """音频播放完成的回调"""
        self.update_status("一段音频播放完成")
        
        # 清除播放器引用
        temp_player = self.current_player
        self.current_player = None
        
        # 确保线程对象能够被垃圾回收
        if temp_player:
            temp_player.deleteLater()
            
        # 播放下一个音频
        self.update_status("尝试播放下一段音频")
        self.play_next_audio()
    
    def stop(self):
        """停止所有TTS处理和播放"""
        self.update_status("正在停止所有TTS处理和播放")
        
        # 停止当前转换
        if self.current_converter:
            self.current_converter.stop()
            
        # 停止下一个并行转换（如果有）
        if self.next_converter:
            self.next_converter.stop()
            
        # 停止当前播放
        if self.current_player:
            self.current_player.stop()
            
        # 清空队列
        while not self.tts_queue.empty():
            self.tts_queue.get()
            
        # 清空音频队列并删除临时文件
        while not self.audio_queue.empty():
            audio_file = self.audio_queue.get()
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
                
        self.is_processing = False
        self.is_playing = False
        
        self.update_status("TTS播放已停止")

class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings.copy()
        self.setWindowTitle("Eidolon Assist Settings")
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs
        tab_widget = QTabWidget()
        
        # API Settings Tab
        api_tab = BlurredWidget()
        api_layout = QFormLayout(api_tab)
        
        self.groq_api_key = QLineEdit(self.settings["api_keys"]["groq"])
        self.groq_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addRow("Groq API Key:", self.groq_api_key)
        
        self.openai_api_key = QLineEdit(self.settings["api_keys"]["openai_compatible"])
        self.openai_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addRow("OpenAI-compatible API Key:", self.openai_api_key)
        
        self.openai_base_url = QLineEdit(self.settings["base_urls"]["openai_compatible"])
        api_layout.addRow("OpenAI-compatible Base URL:", self.openai_base_url)
        
        # Model Settings Tab
        model_tab = BlurredWidget()
        model_layout = QFormLayout(model_tab)
        
        self.whisper_model = QLineEdit(self.settings["models"]["whisper"])
        model_layout.addRow("Whisper Model:", self.whisper_model)
        
        self.llm_model = QLineEdit(self.settings["models"]["llm"])
        model_layout.addRow("LLM Model:", self.llm_model)
        
        self.system_prompt = QTextEdit(self.settings["system_prompt"])
        self.system_prompt.setMinimumHeight(100)
        model_layout.addRow("System Prompt:", self.system_prompt)
        
        # TTS Settings Tab
        tts_tab = BlurredWidget()
        tts_layout = QFormLayout(tts_tab)
        
        self.tts_voice = QComboBox()
        self.tts_voices = [
            "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural",
            "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "ja-JP-NanamiNeural"
        ]
        self.tts_voice.addItems(self.tts_voices)
        current_voice = self.settings["tts_voice"]
        if current_voice in self.tts_voices:
            self.tts_voice.setCurrentText(current_voice)
        tts_layout.addRow("TTS Voice:", self.tts_voice)
        
        # Add TTS Speed slider and input control
        speed_container = QWidget()
        speed_layout = QHBoxLayout(speed_container)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        
        # Get current speed value (remove % and convert to int)
        current_speed_str = self.settings.get("tts_speed", "+0%")
        try:
            current_speed = int(current_speed_str.replace("%", ""))
        except:
            current_speed = 0
            
        # Create slider that goes from -50 to +50
        self.tts_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_speed_slider.setRange(-50, 50)
        self.tts_speed_slider.setValue(current_speed)
        self.tts_speed_slider.setTickInterval(10)
        self.tts_speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        # Create input box
        self.tts_speed_input = QLineEdit(current_speed_str)
        self.tts_speed_input.setFixedWidth(60)
        
        # Connect signals
        self.tts_speed_slider.valueChanged.connect(self.update_speed_input)
        self.tts_speed_input.editingFinished.connect(self.update_speed_slider)
        
        speed_layout.addWidget(self.tts_speed_slider)
        speed_layout.addWidget(self.tts_speed_input)
        
        tts_layout.addRow("TTS Speed:", speed_container)
        
        # Add TTS Volume slider and input control
        volume_container = QWidget()
        volume_layout = QHBoxLayout(volume_container)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        
        # Get current volume value (0.0-1.0)
        current_volume = self.settings.get("tts_volume", 1.0)
        
        # Create slider that goes from 0 to 100
        self.tts_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_volume_slider.setRange(0, 100)
        self.tts_volume_slider.setValue(int(current_volume * 100))
        self.tts_volume_slider.setTickInterval(10)
        self.tts_volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        # Create input box
        self.tts_volume_input = QLineEdit(f"{int(current_volume * 100)}%")
        self.tts_volume_input.setFixedWidth(60)
        
        # Connect signals
        self.tts_volume_slider.valueChanged.connect(self.update_volume_input)
        self.tts_volume_input.editingFinished.connect(self.update_volume_slider)
        
        volume_layout.addWidget(self.tts_volume_slider)
        volume_layout.addWidget(self.tts_volume_input)
        
        tts_layout.addRow("TTS Volume:", volume_container)
        
        # Hotkeys Settings Tab
        hotkeys_tab = BlurredWidget()
        hotkeys_layout = QFormLayout(hotkeys_tab)
        
        self.start_recording_key = QLineEdit(self.settings["hotkeys"]["start_recording"])
        hotkeys_layout.addRow("Start Recording:", self.start_recording_key)
        
        self.stop_recording_key = QLineEdit(self.settings["hotkeys"]["stop_recording"])
        hotkeys_layout.addRow("Stop Recording:", self.stop_recording_key)
        
        self.screenshot_key = QLineEdit(self.settings["hotkeys"]["take_screenshot"])
        hotkeys_layout.addRow("Take Screenshot:", self.screenshot_key)
        
        self.delete_screenshot_key = QLineEdit(self.settings["hotkeys"]["delete_last_screenshot"])
        hotkeys_layout.addRow("Delete Last Screenshot:", self.delete_screenshot_key)
        
        # General Settings Tab
        general_tab = BlurredWidget()
        general_layout = QFormLayout(general_tab)
        
        # Microphone selection
        self.mic_combo = QComboBox()
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))
                self.mic_combo.addItem(f"{i}: {device['name']}")
        
        # Set current microphone
        for i, (idx, _) in enumerate(input_devices):
            if idx == self.settings["microphone_index"]:
                self.mic_combo.setCurrentIndex(i)
                break
        
        general_layout.addRow("Microphone:", self.mic_combo)
        
        # Conversation history
        self.history_limit = QSpinBox()
        self.history_limit.setRange(-1, 100)
        self.history_limit.setValue(self.settings["max_conversation_history"])
        self.history_limit.setSpecialValueText("Unlimited")
        general_layout.addRow("Conversation History Limit:", self.history_limit)
        
        # Add tabs
        tab_widget.addTab(api_tab, "API Keys")
        tab_widget.addTab(model_tab, "Models")
        tab_widget.addTab(tts_tab, "Text-to-Speech")
        tab_widget.addTab(hotkeys_tab, "Hotkeys")
        tab_widget.addTab(general_tab, "General")
        
        layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
    
    def update_speed_input(self, value):
        """Update the speed input box when slider changes"""
        sign = "+" if value >= 0 else ""
        self.tts_speed_input.setText(f"{sign}{value}%")
    
    def update_speed_slider(self):
        """Update the speed slider when input changes"""
        try:
            text = self.tts_speed_input.text().strip()
            # Remove any sign and % character
            value_text = text.replace("+", "").replace("%", "")
            value = int(value_text)
            # Ensure the value is within slider range
            value = max(-50, min(50, value))
            self.tts_speed_slider.setValue(value)
            # Update the text to ensure proper formatting
            sign = "+" if value >= 0 else ""
            self.tts_speed_input.setText(f"{sign}{value}%")
        except ValueError:
            # If invalid input, restore from slider value
            self.update_speed_input(self.tts_speed_slider.value())
    
    def update_volume_input(self, value):
        """Update the volume input box when slider changes"""
        self.tts_volume_input.setText(f"{value}%")
    
    def update_volume_slider(self):
        """Update the volume slider when input changes"""
        try:
            text = self.tts_volume_input.text().strip()
            # Remove % character
            value_text = text.replace("%", "")
            value = int(value_text)
            # Ensure the value is within slider range
            value = max(0, min(100, value))
            self.tts_volume_slider.setValue(value)
            # Update the text to ensure proper formatting
            self.tts_volume_input.setText(f"{value}%")
        except ValueError:
            # If invalid input, restore from slider value
            self.update_volume_input(self.tts_volume_slider.value())
    
    def get_settings(self):
        if self.result() == QDialog.DialogCode.Accepted:
            # Parse microphone index from selection
            mic_text = self.mic_combo.currentText()
            try:
                mic_index = int(mic_text.split(":")[0])
            except:
                mic_index = 0
                
            # Get TTS speed with proper format (ensure it has sign and %)
            speed_value = self.tts_speed_slider.value()
            speed_text = f"+{speed_value}%" if speed_value >= 0 else f"{speed_value}%"
            
            # Get TTS volume (convert from 0-100 to 0.0-1.0)
            volume_value = self.tts_volume_slider.value() / 100.0
            
            return {
                "api_keys": {
                    "groq": self.groq_api_key.text(),
                    "openai_compatible": self.openai_api_key.text()
                },
                "base_urls": {
                    "openai_compatible": self.openai_base_url.text()
                },
                "models": {
                    "whisper": self.whisper_model.text(),
                    "llm": self.llm_model.text()
                },
                "tts_voice": self.tts_voice.currentText(),
                "tts_speed": speed_text,  # Save TTS speed with proper format
                "tts_volume": volume_value,  # Save TTS volume as float between 0.0-1.0
                "hotkeys": {
                    "start_recording": self.start_recording_key.text(),
                    "stop_recording": self.stop_recording_key.text(),
                    "take_screenshot": self.screenshot_key.text(),
                    "delete_last_screenshot": self.delete_screenshot_key.text()
                },
                "microphone_index": mic_index,
                "max_conversation_history": self.history_limit.value(),
                "system_prompt": self.system_prompt.toPlainText()
            }
        return self.settings

class EidolonAssistApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Load settings
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        self.settings = self.load_settings()
        
        # Initialize state variables
        self.current_audio_file = None
        self.screenshot_paths = []
        self.conversation_history = []
        self.is_recording = False
        self.recorder = None
        self.transcriber = None
        self.llm_thread = None
        self.tts_processor = ParallelTTSProcessor(self.settings)
        self.tts_processor.set_status_callback(self.update_status)
        
        # Create UI
        self.init_ui()
        
        # Setup global hotkeys
        self.setup_hotkeys()
    
    def load_settings(self):
        """Load settings from file or use defaults"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Ensure all default settings keys exist
                    for key in DEFAULT_SETTINGS:
                        if key not in settings:
                            settings[key] = DEFAULT_SETTINGS[key]
                        elif isinstance(DEFAULT_SETTINGS[key], dict):
                            for subkey in DEFAULT_SETTINGS[key]:
                                if subkey not in settings[key]:
                                    settings[key][subkey] = DEFAULT_SETTINGS[key][subkey]
                    return settings
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
        
        return DEFAULT_SETTINGS.copy()
    
    def save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Eidolon Assist")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Conversation and Screenshots
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Conversation area
        conversation_container = BlurredWidget()
        conversation_layout = QVBoxLayout(conversation_container)
        
        # 添加对话标题和新对话按钮的水平布局
        conversation_header = QHBoxLayout()
        
        conversation_label = QLabel("Conversation")
        conversation_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        conversation_header.addWidget(conversation_label)
        
        # 添加新对话按钮
        new_conversation_btn = QPushButton("新对话")
        new_conversation_btn.clicked.connect(self.clear_conversation_history)
        conversation_header.addWidget(new_conversation_btn)
        
        self.conversation_text = QTextEdit()
        self.conversation_text.setReadOnly(True)
        self.conversation_text.setStyleSheet("background-color: rgba(255, 255, 255, 150); border-radius: 10px; padding: 10px;")
        
        conversation_layout.addLayout(conversation_header)
        conversation_layout.addWidget(self.conversation_text)
        
        # Screenshots area
        screenshots_container = BlurredWidget()
        screenshots_layout = QVBoxLayout(screenshots_container)
        
        screenshots_header = QHBoxLayout()
        screenshots_label = QLabel("Screenshots")
        screenshots_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        screenshot_btn = QPushButton("Take Screenshot")
        screenshot_btn.clicked.connect(self.take_screenshot)
        
        clear_screenshots_btn = QPushButton("Clear All")
        clear_screenshots_btn.clicked.connect(self.clear_screenshots)
        
        screenshots_header.addWidget(screenshots_label)
        screenshots_header.addWidget(screenshot_btn)
        screenshots_header.addWidget(clear_screenshots_btn)
        
        self.screenshots_scroll = QScrollArea()
        self.screenshots_scroll.setWidgetResizable(True)
        self.screenshots_widget = QWidget()
        self.screenshots_layout = QVBoxLayout(self.screenshots_widget)
        self.screenshots_scroll.setWidget(self.screenshots_widget)
        self.screenshots_scroll.setStyleSheet("background-color: rgba(255, 255, 255, 150); border-radius: 10px;")
        
        screenshots_layout.addLayout(screenshots_header)
        screenshots_layout.addWidget(self.screenshots_scroll)
        
        # Add conversation and screenshots to top layout
        top_layout.addWidget(conversation_container, 2)  # Larger proportion for conversation
        top_layout.addWidget(screenshots_container, 1)   # Smaller proportion for screenshots
        
        # Bottom section - Controls
        bottom_widget = BlurredWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        
        # Status bar
        status_container = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_container.addWidget(self.status_label)
        status_container.addStretch()
        
        # Recording controls
        controls_container = QHBoxLayout()
        
        self.record_btn = QPushButton("Record (Ctrl+[)")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.stop_tts_btn = QPushButton("Stop TTS (Ctrl+')")
        self.stop_tts_btn.clicked.connect(self.stop_tts)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        
        controls_container.addWidget(self.record_btn)
        controls_container.addWidget(self.stop_tts_btn)
        controls_container.addWidget(self.settings_btn)
        
        # Text input area
        input_container = QHBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.setMaximumHeight(100)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_text_message)
        
        input_container.addWidget(self.text_input)
        input_container.addWidget(self.send_btn)
        
        # Add all controls to bottom layout
        bottom_layout.addLayout(status_container)
        bottom_layout.addLayout(controls_container)
        bottom_layout.addLayout(input_container)
        
        # Add top and bottom sections to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([400, 200])  # Initial sizes
        
        main_layout.addWidget(splitter)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QLabel {
                color: #333;
            }
        """)
    
    def setup_hotkeys(self):
        """Setup global hotkeys for actions"""
        try:
            # Remove any existing hotkeys first
            keyboard.unhook_all()
            
            # Setup new hotkeys from settings
            keyboard.add_hotkey(self.settings["hotkeys"]["start_recording"], lambda: self.safe_thread_call(self.start_recording))
            keyboard.add_hotkey(self.settings["hotkeys"]["stop_recording"], lambda: self.safe_thread_call(self.stop_recording))
            keyboard.add_hotkey(self.settings["hotkeys"]["take_screenshot"], lambda: self.safe_thread_call(self.take_screenshot))
            keyboard.add_hotkey(self.settings["hotkeys"]["delete_last_screenshot"], lambda: self.safe_thread_call(self.delete_last_screenshot))
            keyboard.add_hotkey(self.settings["hotkeys"].get("stop_tts", "ctrl+'"), lambda: self.safe_thread_call(self.stop_tts))
        except Exception as e:
            self.update_status(f"Error setting up hotkeys: {str(e)}")
    
    def safe_thread_call(self, func):
        """Safely call a function from any thread by using QTimer to ensure it runs in the main thread"""
        QTimer.singleShot(0, func)
    
    def open_settings(self):
        """Open the settings dialog"""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            # Get the new settings
            new_settings = dialog.get_settings()
            
            # Check if hotkeys changed
            hotkeys_changed = (self.settings["hotkeys"] != new_settings["hotkeys"])
            
            # Update settings
            self.settings = new_settings
            self.save_settings()
            
            # Reinitialize hotkeys if they changed
            if hotkeys_changed:
                self.setup_hotkeys()
            
            self.update_status("Settings updated")
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        # 先停止正在进行的TTS播放
        self.stop_tts()
            
        self.is_recording = True
        self.record_btn.setText("Stop Recording (Ctrl+])")
        
        self.recorder = AudioRecorder(self.settings)
        self.recorder.status_update.connect(self.update_status)
        self.recorder.finished.connect(self.recording_finished)
        self.recorder.start()
    
    def stop_recording(self):
        """Stop audio recording and process it"""
        if not self.is_recording:
            return
            
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def recording_finished(self, audio_file):
        """Handle the recorded audio file"""
        self.is_recording = False
        self.record_btn.setText("Record (Ctrl+[)")
        
        if not audio_file:
            self.update_status("Recording failed or no audio captured")
            return
            
        self.current_audio_file = audio_file
        self.update_status("Transcribing audio...")
        
        # Start transcription
        self.transcriber = TranscriptionThread(audio_file, self.settings)
        self.transcriber.status_update.connect(self.update_status)
        self.transcriber.finished.connect(self.transcription_finished)
        self.transcriber.start()
    
    def transcription_finished(self, transcription_text):
        """Handle the transcribed text"""
        if not transcription_text:
            self.update_status("Transcription failed or no text recognized")
            return
            
        # Add transcription to conversation
        self.add_to_conversation("You", transcription_text)
        
        # Create messages for the LLM
        messages = self.create_messages_for_llm(transcription_text)
        
        # Start LLM processing
        self.llm_thread = LLMThread(messages, self.settings, self.screenshot_paths)
        self.llm_thread.status_update.connect(self.update_status)
        self.llm_thread.token_received.connect(self.append_assistant_response)
        self.llm_thread.sentence_complete.connect(self.speak_text)
        self.llm_thread.finished.connect(self.llm_response_finished)
        self.llm_thread.start()
    
    def send_text_message(self):
        """Send a text message from the input field"""
        message = self.text_input.toPlainText().strip()
        if not message:
            return
            
        # Clear the input field
        self.text_input.clear()
        
        # Add message to conversation
        self.add_to_conversation("You", message)
        
        # Create messages for the LLM
        messages = self.create_messages_for_llm(message)
        
        # Start LLM processing
        self.llm_thread = LLMThread(messages, self.settings, self.screenshot_paths)
        self.llm_thread.status_update.connect(self.update_status)
        self.llm_thread.token_received.connect(self.append_assistant_response)
        self.llm_thread.sentence_complete.connect(self.speak_text)
        self.llm_thread.finished.connect(self.llm_response_finished)
        self.llm_thread.start()
    
    def create_messages_for_llm(self, user_message):
        """Create messages for the language model with conversation history"""
        messages = []
        
        # Add system message
        if self.settings["system_prompt"]:
            messages.append({"role": "system", "content": self.settings["system_prompt"]})
        
        # Add conversation history with limit
        max_history = self.settings["max_conversation_history"]
        history = self.conversation_history
        
        if max_history > 0:
            history = history[-max_history:]
        
        for entry in history:
            role = "assistant" if entry["speaker"] == "Assistant" else "user"
            messages.append({"role": role, "content": entry["message"]})
            
        # Add new user message if not already in history
        if history and history[-1]["speaker"] == "You" and history[-1]["message"] == user_message:
            # Already in history
            pass
        else:
            messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def append_assistant_response(self, token):
        """Append a token from the streaming response"""
        # Check if there's an existing assistant message in the conversation
        if self.conversation_history and self.conversation_history[-1]["speaker"] == "Assistant":
            # Append to existing message
            self.conversation_history[-1]["message"] += token
            
            # Update the conversation display
            cursor = self.conversation_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(token)
            self.conversation_text.setTextCursor(cursor)
        else:
            # Add a new assistant message
            self.add_to_conversation("Assistant", token, append=False)
    
    def llm_response_finished(self, full_response):
        """Handle the complete LLM response"""
        if not full_response:
            return
            
        # We no longer need to speak the full response here
        # The sentences are already being spoken via the sentence_complete signal
        pass
    
    def add_to_conversation(self, speaker, message, append=True):
        """Add a message to the conversation history and display"""
        # Add to history
        if append or not self.conversation_history or self.conversation_history[-1]["speaker"] != speaker:
            self.conversation_history.append({
                "speaker": speaker,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Append to the existing message from the same speaker
            self.conversation_history[-1]["message"] += message
        
        # Update the display
        formatted_text = ""
        
        for entry in self.conversation_history:
            speaker_style = "color: #0066cc; font-weight: bold;" if entry["speaker"] == "Assistant" else "color: #006600; font-weight: bold;"
            formatted_text += f'<p><span style="{speaker_style}">{entry["speaker"]}:</span> {entry["message"]}</p>'
        
        self.conversation_text.setHtml(formatted_text)
        
        # Scroll to the bottom
        self.conversation_text.verticalScrollBar().setValue(self.conversation_text.verticalScrollBar().maximum())
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        self.tts_processor.add_text(text)
    
    def stop_tts(self):
        """Stop the current TTS playback and clear the queue"""
        self.tts_processor.stop()
    
    def take_screenshot(self):
        """Take a screenshot of the entire screen"""
        try:
            self.update_status("Taking screenshot...")
            
            # Take screenshot
            screenshot = ImageGrab.grab()
            
            # Save to temp file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_file = os.path.join(tempfile.gettempdir(), f"eidolon_screenshot_{timestamp}.png")
            screenshot.save(screenshot_file)
            
            # Add to screenshot list
            self.screenshot_paths.append(screenshot_file)
            
            # Update UI
            self.add_screenshot_to_ui(screenshot_file)
            
            self.update_status(f"Screenshot saved: {os.path.basename(screenshot_file)}")
            
        except Exception as e:
            self.update_status(f"Screenshot error: {str(e)}")
    
    def add_screenshot_to_ui(self, file_path):
        """Add a screenshot thumbnail to the UI"""
        try:
            # Container for screenshot and delete button
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            # Load image and create thumbnail
            pixmap = QPixmap(file_path)
            thumbnail = pixmap.scaled(QSize(200, 150), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            # Create image label
            image_label = QLabel()
            image_label.setPixmap(thumbnail)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("background-color: white; border: 1px solid #cccccc; border-radius: 5px;")
            
            # Create delete button
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda: self.delete_screenshot(file_path))
            
            # Add to layout
            container_layout.addWidget(image_label)
            container_layout.addWidget(delete_btn)
            
            # Insert at the top of the layout
            self.screenshots_layout.insertWidget(0, container)
            
        except Exception as e:
            self.update_status(f"Error adding screenshot to UI: {str(e)}")
    
    def delete_screenshot(self, file_path):
        """Delete a specific screenshot"""
        try:
            # Remove from list
            if file_path in self.screenshot_paths:
                self.screenshot_paths.remove(file_path)
                
                # Try to delete the file
                try:
                    os.remove(file_path)
                except:
                    pass
                
                # Refresh the screenshots UI
                self.refresh_screenshots_ui()
                
                self.update_status(f"Screenshot deleted: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.update_status(f"Error deleting screenshot: {str(e)}")
    
    def delete_last_screenshot(self):
        """Delete the last screenshot"""
        if self.screenshot_paths:
            last_screenshot = self.screenshot_paths[-1]
            self.delete_screenshot(last_screenshot)
    
    def clear_screenshots(self):
        """Clear all screenshots"""
        try:
            # Delete all screenshot files
            for file_path in self.screenshot_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            # Clear list
            self.screenshot_paths = []
            
            # Refresh UI
            self.refresh_screenshots_ui()
            
            self.update_status("All screenshots cleared")
            
        except Exception as e:
            self.update_status(f"Error clearing screenshots: {str(e)}")
    
    def refresh_screenshots_ui(self):
        """Refresh the screenshots display"""
        # Clear the current screenshots layout
        while self.screenshots_layout.count():
            widget = self.screenshots_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        
        # Add all screenshots back
        for file_path in reversed(self.screenshot_paths):
            self.add_screenshot_to_ui(file_path)
    
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_label.setText(message)
        print(message)  # Also log to console for debugging
    
    def clear_conversation_history(self):
        """清除所有对话历史记录，开始新对话"""
        # 停止任何正在进行的 TTS 播放
        self.stop_tts()
        
        # 清除历史记录数组
        self.conversation_history = []
        
        # 清除界面显示
        self.conversation_text.clear()
        
        # 更新状态
        self.update_status("已开始新对话")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any running threads
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            
        if self.tts_processor:
            self.tts_processor.stop()
        
        # Remove any existing LLM thread
        if self.llm_thread and self.llm_thread.isRunning():
            self.llm_thread.wait()
            
        # Remove any existing transcriber thread
        if self.transcriber and self.transcriber.isRunning():
            self.transcriber.wait()
        
        # Clean up temporary files
        try:
            if self.current_audio_file and os.path.exists(self.current_audio_file):
                os.remove(self.current_audio_file)
                
            for file_path in self.screenshot_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
        except:
            pass
            
        # Unhook all hotkeys
        keyboard.unhook_all()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = EidolonAssistApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()