"""Audio cue and sound effects manager."""

from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("audio_manager")


class AudioManager:
    """Manages audio chime cues and sound effects for navigation warnings."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def play_warning_beep(self) -> None:
        """Plays a warning audio tone."""
        if not self.enabled:
            return
        logger.info("[AUDIO CUE] Warning beep tone played.")

    def play_critical_alarm(self) -> None:
        """Plays a critical alarm tone."""
        if not self.enabled:
            return
        logger.info("[AUDIO CUE] Critical alarm tone played.")
