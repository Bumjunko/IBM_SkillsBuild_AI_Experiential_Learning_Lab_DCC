"""Watson Speech to Text WebSocket client wrapper.

Receives audio chunks from the browser (via FastAPI WebSocket),
forwards them to Watson STT, and returns partial/final transcripts.
"""

import json
import logging
import asyncio
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from commcopilot.config import WATSON_STT_API_KEY, WATSON_STT_URL

logger = logging.getLogger(__name__)


class STTCallback(RecognizeCallback):
    """Callback handler for Watson STT streaming results."""

    def __init__(self, on_transcript):
        super().__init__()
        self.on_transcript = on_transcript

    def on_data(self, data):
        results = data.get("results", [])
        for result in results:
            transcript = result["alternatives"][0]["transcript"]
            is_final = result.get("final", False)
            self.on_transcript(transcript, is_final)

    def on_error(self, error):
        logger.error("Watson STT error: %s", error)

    def on_close(self):
        logger.info("Watson STT connection closed")


class WatsonSTTClient:
    """Manages a Watson STT streaming session."""

    def __init__(self, on_transcript_callback):
        """
        Args:
            on_transcript_callback: async function(transcript: str, is_final: bool)
        """
        self.on_transcript_callback = on_transcript_callback
        self._authenticator = IAMAuthenticator(WATSON_STT_API_KEY)
        self._stt = SpeechToTextV1(authenticator=self._authenticator)
        self._stt.set_service_url(WATSON_STT_URL)
        self._audio_source = None
        self._recognize_thread = None

    def start(self):
        """Start the STT streaming session."""
        self._audio_source = AudioSource(is_recording=True)

        def sync_callback(transcript, is_final):
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.ensure_future,
                self.on_transcript_callback(transcript, is_final),
            )

        callback = STTCallback(on_transcript=sync_callback)

        self._recognize_thread = self._stt.recognize_using_websocket(
            audio=self._audio_source,
            content_type="audio/webm;codecs=opus",
            recognize_callback=callback,
            model="en-US_BroadbandModel",
            interim_results=True,
        )
        logger.info("Watson STT streaming started")

    def send_audio(self, audio_data: bytes):
        """Send an audio chunk to Watson STT."""
        if self._audio_source:
            self._audio_source.input.put(audio_data)

    def stop(self):
        """Stop the STT streaming session."""
        if self._audio_source:
            self._audio_source.completed_recording()
            self._audio_source = None
        logger.info("Watson STT streaming stopped")
