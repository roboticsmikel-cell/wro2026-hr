# ALZONA backend (main.py) for Render, or any Docker host.
#
# A server has no webcam, printer or Arduino, so this runs ALZONA without them:
# chat (Gemini + the knowledge files), the ElevenLabs/Gemini voice, Baybayin
# images on screen, and the harmony data. The camera, greetings, coin reading
# and printing need the robot's own laptop - run start-5175.bat there.
#
# Python 3.12: mediapipe 0.10.21 (pinned in requirements.txt) has no wheels
# for 3.13, and main.py uses its mp.solutions API, which newer releases drop.
FROM python:3.12-slim-bookworm

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV and mediapipe load libGL/glib on import even with no display;
# sounddevice (a mediapipe dependency) wants PortAudio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code-only change reuses this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# No camera and no printer on a server (see main.py: CAMERA_ENABLED,
# BAYBAYIN_AUTOPRINT). Render's own env vars override these if set.
ENV CAMERA_INDEX=off \
    BAYBAYIN_AUTOPRINT=0

# main.py reads PORT (Render sets it) and falls back to 5002.
EXPOSE 5002
CMD ["python", "main.py"]
