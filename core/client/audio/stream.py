# coding: utf-8
"""
音频流管理模块

提供 AudioStreamManager 类用于管理音频输入流，包括流的创建、
启动、停止和设备检测。

关键设计：把「一次性启动、失败即死」改成「持续守护、设备/会话恢复即恢复」。

RDP 会话切换、热插拔、睡眠唤醒、独占冲突等场景下，音频端点会瞬间失效，
表现有两种：
  1. 设备直接消失（query_devices 抛 PortAudioError，流结束回调触发）；
  2. 流「还活着、回调还在」，但送进来的全是数字静音（实测 -91 dB）。
为此本模块：
  - 不再因取设备失败而 input()/sys.exit() 自杀；
  - 后台守护线程退避重试，设备/会话恢复后自动重连；
  - 轮询 Windows 会话连接状态，检测到会话重新激活（如 RDP 断开后返回）就重建流；
  - 整段录音若全是数字静音，判定采集失效并重建流。
"""

from __future__ import annotations

import sys
import time
import threading
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import sounddevice as sd

from core.client.state import console
from . import logger

if TYPE_CHECKING:
    from core.client.state import ClientState
    from ..app import CapsWriterClient


# 重试退避参数（秒）
RETRY_MIN_DELAY = 1.0
RETRY_MAX_DELAY = 30.0
RETRY_BACKOFF = 2.0
# 存活探针：录音中超过该时长没有回调，判定流卡死强制重启
LIVENESS_TIMEOUT = 15.0
# 会话连接状态轮询间隔（秒）
SESSION_POLL_INTERVAL = 2.0
# 判定「有声音」的峰值阈值（float32 归一化，真实麦克风底噪通常远大于此）
SOUND_PEAK_THRESHOLD = 1e-4
# 判定「整段静音失效」所需的最少回调块数（0.05s/块，20 块≈1s）
SILENT_BLOCK_MIN = 20

# WTSConnectState：0 = WTSActive（会话处于活动状态）
_WTS_ACTIVE = 0
_WTS_CURRENT_SESSION = 0xFFFFFFFF
_WTS_CONNECT_STATE = 8


class AudioStreamManager:
    """
    音频流管理器

    负责管理音频输入流的生命周期，包括：
    - 检测和选择音频设备
    - 创建和启动音频流
    - 处理音频数据回调
    - 流的重启、关闭，以及设备/会话恢复后的后台自动重连

    Attributes:
        state: 客户端状态实例
        sample_rate: 采样率（默认 48000Hz）
        block_duration: 每个数据块的时长（秒，默认 0.05s）
    """

    SAMPLE_RATE = 48000
    BLOCK_DURATION = 0.05  # 50ms

    def __init__(self, app: CapsWriterClient):
        """
        初始化音频流管理器

        Args:
            app: 客户端 App 实例
        """
        self.app = app
        self._channels = 1
        self._running = False    # 流当前是否处于运行态
        self._enabled = True     # 是否还需要流（app 关闭时置 False，停止后台守护）
        self._supervising = False  # 守护线程是否在跑（防止并发起多个）
        self._supervisor_lock = threading.Lock()
        self._supervisor_thread: Optional[threading.Thread] = None
        self._session_thread: Optional[threading.Thread] = None
        self._last_hint_ts = 0.0   # 上一次打印 -9999 权限提示的时间（限频）
        self._last_cb_ts = 0.0     # 上一次收到音频回调的时间（存活探针）

        # 整段静音看门狗用的录音期统计
        self._was_recording = False
        self._rec_blocks = 0
        self._rec_zero_blocks = 0
        self._rec_had_sound = False
        self._dead_audio = False   # 回调线程置位，守护线程据此重建流

    @property
    def state(self) -> ClientState:
        """快捷访问状态单例"""
        return self.app.state

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags
    ) -> None:
        """
        音频数据回调函数

        当音频流接收到新数据时调用，将数据放入异步队列中。
        """
        # 存活探针：无论是否处于录音态都记录，用于检测「流在跑但不再回调」的卡死
        self._last_cb_ts = time.time()

        rec = bool(self.state.recording)
        # 录音开始：重置本段统计
        if rec and not self._was_recording:
            self._rec_blocks = 0
            self._rec_zero_blocks = 0
            self._rec_had_sound = False

        if rec:
            self._rec_blocks += 1
            try:
                peak = float(np.max(np.abs(indata))) if indata.size else 0.0
            except Exception:
                peak = 1.0  # 计算异常时按「有声」处理，避免误判
            if peak > SOUND_PEAK_THRESHOLD:
                self._rec_had_sound = True
            else:
                self._rec_zero_blocks += 1
        elif self._was_recording:
            # 一次录音刚结束：整段全是数字静音 → 采集已失效，标记重建
            if (self._rec_blocks >= SILENT_BLOCK_MIN
                    and not self._rec_had_sound
                    and self._rec_zero_blocks >= self._rec_blocks):
                logger.warning(
                    f"整段录音全为数字静音（{self._rec_blocks} 块），判定音频流已失效，准备重建"
                )
                self._dead_audio = True

        self._was_recording = rec

        # 只在录音状态时处理数据
        if not rec:
            return

        import asyncio

        # 将数据放入队列
        if self.app.loop and self.state.queue_in:
            asyncio.run_coroutine_threadsafe(
                self.state.queue_in.put({
                    'type': 'data',
                    'time': time.time(),
                    'data': indata.copy(),
                }),
                self.app.loop
            )

    def _on_stream_finished(self) -> None:
        """音频流结束回调（由 PortAudio 在工作线程中调用）"""
        if not threading.main_thread().is_alive():
            return
        if not self._running:
            return

        logger.info("音频流意外结束，正在尝试重启…")
        self._running = False
        self.state.stream = None
        # 由守护线程负责退避重试，这里只触发，不在回调线程里做阻塞/重载
        self._ensure_supervisor()

    # ----------------------------------------------------------------
    # 设备探测
    # ----------------------------------------------------------------
    @staticmethod
    def _safe_name(device) -> str:
        """安全读取设备名，跳过编码问题"""
        try:
            return device.get('name', '未知设备')
        except (UnicodeDecodeError, AttributeError):
            return '未知设备'

    def _pick_input_device(self) -> Tuple[Optional[dict], str, int]:
        """
        选出可用的输入设备。

        Returns:
            (device, name, channels)
            - device 为 None 表示当前没有任何可用输入设备（需后台重试）
            - 优先用系统默认输入端点；默认失效时回退到任意可用输入，
              主机 API 优先级 WASAPI > DirectSound > MME
        """
        # 1) 优先系统默认输入端点
        try:
            device = sd.query_devices(kind='input')
            if device is not None and device.get('max_input_channels', 0) > 0:
                return device, self._safe_name(device), min(2, device['max_input_channels'])
        except (sd.PortAudioError, UnicodeDecodeError) as e:
            logger.debug(f"取默认输入设备失败: {e}")
        except Exception as e:  # 其它未知异常不能阻断流程
            logger.debug(f"取默认输入设备异常: {e}")

        # 2) 默认失效，枚举全部输入设备找一个可用的
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            logger.debug(f"枚举音频设备失败: {e}")
            return None, '未知设备', 1

        candidates = [
            (i, d) for i, d in enumerate(devices)
            if d.get('max_input_channels', 0) > 0
        ]
        if not candidates:
            return None, '未知设备', 1

        rank_order = {'Windows WASAPI': 0, 'Windows DirectSound': 1, 'MME': 2}

        def _rank(d) -> int:
            try:
                name = hostapis[d['hostapi']]['name']
            except Exception:
                name = ''
            return rank_order.get(name, 3)

        candidates.sort(key=lambda c: _rank(c[1]))
        _i, d = candidates[0]
        return d, self._safe_name(d), min(2, d['max_input_channels'])

    # ----------------------------------------------------------------
    # 启动 / 守护 / 会话检测
    # ----------------------------------------------------------------
    def start(self) -> Optional[sd.InputStream]:
        """
        尝试创建并启动音频流（单次）。

        Returns:
            创建的音频输入流，失败（设备不可用/被占）返回 None。
            无论成功失败都会确保后台守护线程在跑；绝不退出进程。

        注意：本方法可能在回调线程、托盘线程或主线程被调用，
              因此不得执行 input()/sys.exit() 这类会阻塞或杀线程的操作。
        """
        if self._running:
            logger.debug("音频流已在运行，跳过启动")
            return self.state.stream

        device, device_name, channels = self._pick_input_device()
        if device is None:
            logger.error(
                "未找到麦克风设备（可能处于 RDP 会话切换、麦克风被禁用或独占占用，"
                "将在后台自动重试）"
            )
            self._ensure_supervisor()
            return None

        self._channels = channels
        console.print(
            f'使用默认音频设备：[italic]{device_name}，声道数：{self._channels}',
            end='\n\n'
        )
        logger.info(f"找到音频设备: {device_name}, 声道数: {self._channels}")

        # 创建音频流
        try:
            stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=int(self.BLOCK_DURATION * self.SAMPLE_RATE),
                device=None,
                dtype="float32",
                channels=self._channels,
                callback=self._audio_callback,
                finished_callback=self._on_stream_finished,
            )
            stream.start()

            self.state.stream = stream
            self._running = True
            logger.debug(
                f"音频流已启动: 采样率={self.SAMPLE_RATE}, "
                f"块大小={int(self.BLOCK_DURATION * self.SAMPLE_RATE)}"
            )
            # 确保有守护线程做存活探针 / 静音检测 / 会话检测
            self._ensure_supervisor()
            self._start_session_watcher()
            return stream

        except sd.PortAudioError as e:
            logger.error(f"创建音频流失败: {e}", exc_info=True)
            if '-9999' in str(e):
                now = time.time()
                if now - self._last_hint_ts > 60:   # 限频：每分钟最多一次
                    self._last_hint_ts = now
                    console.print("""
[bold red]检测到麦克风被占用或权限异常（错误码 -9999）[/bold red]
请尝试以下解决方案：

  1. 设置 > 隐私和安全性 > 麦克风，将「允许桌面应用访问麦克风」打开
  2. 状态栏右下角音量图标 > 右键菜单 > 声音 > 麦克风的属性，关闭「允许应用程序独占控制该设备」
  3. 状态栏右下角音量图标 > 右键菜单 > 声音 > 麦克风的属性，关闭「增强效果」
""")
            self._ensure_supervisor()
            return None
        except Exception as e:
            logger.error(f"创建音频流失败: {e}", exc_info=True)
            self._ensure_supervisor()
            return None

    def _ensure_supervisor(self) -> None:
        """确保后台守护线程在跑（幂等：同时只允许一个）"""
        if not self._enabled:
            return
        with self._supervisor_lock:
            if self._supervising:
                return
            self._supervising = True
        t = threading.Thread(target=self._supervisor, name='audio-supervisor', daemon=True)
        self._supervisor_thread = t
        t.start()

    def _supervisor(self) -> None:
        """
        后台守护线程：
        - 健康态（流在跑）：做存活探针 + 整段静音检测，异常则重建；
        - 异常态（流没跑）：按指数退避重载 PortAudio 并重试，设备/会话恢复后自动重连。
        """
        logger.info("启动音频守护线程")
        delay = RETRY_MIN_DELAY
        attempt = 0
        try:
            while self._enabled:
                if self._running:
                    # 回调线程判定整段静音失效 → 这里重建
                    if self._dead_audio:
                        self._dead_audio = False
                        logger.warning("检测到整段静音录音，重建音频流以恢复采集")
                        self._force_close()
                        continue
                    self._liveness_check()
                    time.sleep(RETRY_MIN_DELAY)
                    continue

                # —— 异常态：流没跑，退避后重试 ——
                time.sleep(delay)
                if not self._enabled:
                    break
                attempt += 1
                logger.info(f"重试连接麦克风（第 {attempt} 次，间隔 {delay:.0f}s）…")
                self._reload_portaudio()
                delay = min(delay * RETRY_BACKOFF, RETRY_MAX_DELAY)
                self.start()   # 成功则 _running=True，下一轮进入健康分支
        finally:
            with self._supervisor_lock:
                self._supervising = False

    def _liveness_check(self) -> None:
        """流声称在跑但录音中长时间没有回调，判定卡死并强制重启。"""
        if not (self.state.recording and self._last_cb_ts > 0):
            return
        if time.time() - self._last_cb_ts > LIVENESS_TIMEOUT:
            logger.warning("音频流疑似卡死（录音中长时间无回调），强制重启")
            self._force_close()

    # ----------------------------------------------------------------
    # 会话连接状态检测（RDP 断开/返回等）
    # ----------------------------------------------------------------
    def _start_session_watcher(self) -> None:
        """启动会话连接状态轮询线程（幂等）"""
        if self._session_thread is not None and self._session_thread.is_alive():
            return
        if sys.platform != 'win32':
            return
        t = threading.Thread(target=self._session_watcher, name='audio-session-watch', daemon=True)
        self._session_thread = t
        t.start()

    def _session_watcher(self) -> None:
        """
        轮询当前 Windows 会话的连接状态。当从「非活动」（如 RDP 占用导致的
        WTSDisconnected/WTSConnected）切回 WTSActive 时，说明用户回到了本会话，
        音频端点已变化，主动重建音频流。
        """
        prev = None
        while self._enabled:
            state = self._query_connect_state()
            if state is not None:
                if prev is not None and prev != _WTS_ACTIVE and state == _WTS_ACTIVE:
                    logger.info("检测到会话重新激活（RDP 断开后返回控制台等），重建音频流")
                    self._force_close()   # 由守护线程负责重连
                prev = state
            time.sleep(SESSION_POLL_INTERVAL)

    @staticmethod
    def _query_connect_state() -> Optional[int]:
        """查询当前会话的 WTSConnectState；失败返回 None（不抛异常）。"""
        try:
            import ctypes
            from ctypes import wintypes
            wtsapi = ctypes.WinDLL('wtsapi32.dll')
            buf = ctypes.c_void_p()
            nbytes = wintypes.DWORD()
            ok = wtsapi.WTSQuerySessionInformationW(
                wintypes.HANDLE(0),
                wintypes.DWORD(_WTS_CURRENT_SESSION),
                ctypes.c_int(_WTS_CONNECT_STATE),
                ctypes.byref(buf),
                ctypes.byref(nbytes),
            )
            if not ok or not buf.value:
                return None
            state = ctypes.cast(buf, ctypes.POINTER(ctypes.c_int)).contents.value
            wtsapi.WTSFreeMemory(buf)
            return int(state)
        except Exception:
            return None

    # ----------------------------------------------------------------
    # 关闭 / 重开
    # ----------------------------------------------------------------
    def _force_close(self) -> None:
        """关闭当前流但不改变 _enabled（仍应继续自动重连）。"""
        self._running = False
        if self.state.stream is not None:
            try:
                self.state.stream.close()
            except Exception as e:
                logger.debug(f"强制关闭音频流时出错: {e}")
            finally:
                self.state.stream = None

    @staticmethod
    def _reload_portaudio() -> None:
        """重载 PortAudio 以刷新设备列表（沿用上游 reopen() 的私有 API）。"""
        try:
            sd._terminate()
            sd._ffi.dlclose(sd._lib)
            sd._lib = sd._ffi.dlopen(sd._libname)
            sd._initialize()
        except Exception as e:
            logger.warning(f"重载 PortAudio 时发生警告: {e}")

    def stop(self) -> None:
        """停止音频子系统（app 关闭时调用）：关闭流并停止后台守护。"""
        logger.info("正在停止音频子系统")
        self._enabled = False
        self._force_close()

    def reopen(self) -> Optional[sd.InputStream]:
        """
        重新启动音频流（托盘「♻️ 重开音频」调用）。

        关闭当前流并调度后台重连；不会退出进程，设备就绪后自动恢复。
        """
        logger.info("手动重启音频流…")
        self._force_close()
        self._ensure_supervisor()
        return None
