"""Text-to-Speech (TTS) voice notification manager using pyttsx3."""

from typing import Optional
import threading
import queue
import time

from ..utils.logger import get_logger

logger = get_logger("voice_alert")


class VoiceAlertManager:
    """Provides thread-safe non-blocking Text-to-Speech audio feedback for visually impaired users.

    Fixes pyttsx3 'run loop already started' errors by managing speech requests sequentially
    via a dedicated background thread and thread-safe Queue.
    """

    def __init__(
        self,
        enabled: bool = True,
        speech_rate: int = 160,
        volume: float = 0.9,
    ) -> None:
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.volume = volume
        self.last_spoken_text: str = ""
        self._speech_queue: queue.Queue = queue.Queue(maxsize=5)
        self._worker_thread: Optional[threading.Thread] = None
        self._running: bool = False

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        """Starts the dedicated TTS background thread worker."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self._worker_thread.start()

    def _tts_worker_loop(self) -> None:
        """Background thread worker handling pyttsx3 speech synthesis sequentially."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
            logger.info("pyttsx3 Text-to-Speech engine initialized in background thread worker.")
        except Exception as e:
            logger.warning(f"pyttsx3 initialization unavailable ({e}). Voice alerts running in log fallback mode.")
            engine = None

        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:  # Stop signal
                break

            logger.info(f"[VOICE ANNOUNCEMENT]: '{text}'")
            self.last_spoken_text = text

            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    logger.error(f"Error during pyttsx3 playback: {e}")
                    # Attempt engine re-initialization if engine crashed
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty("rate", self.speech_rate)
                        engine.setProperty("volume", self.volume)
                    except Exception:
                        engine = None

            self._speech_queue.task_done()

    def speak(self, text: str, non_blocking: bool = True) -> None:
        """Enqueues a text guidance string to be spoken without blocking the main video pipeline.

        Args:
            text: Message string to be announced.
            non_blocking: Ignored (always non-blocking via background queue worker).
        """
        if not self.enabled or not text:
            return

        # Empty older queued messages to prevent voice alert lag behind real-time camera feed
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                break

        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self) -> None:
        """Stops the TTS worker thread cleanly."""
        self._running = False
        if self._speech_queue is not None:
            try:
                self._speech_queue.put_nowait(None)
            except Exception:
                pass
