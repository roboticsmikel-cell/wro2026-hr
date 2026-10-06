import { useEffect, useRef, useState } from 'react'

// The console's OWN camera, for a backend that has none (/state reports
// camera: "browser" — CAMERA_INDEX=off, e.g. on Render).
//
// Shown live in the preview box, and a snapshot is uploaded to /frame twice a
// second. The backend does with each one what the laptop's camera loop does:
// face presence and age for the greeting, and it keeps the latest one for coin
// identification. A frame is only sent once the previous upload has answered,
// so a slow connection drops frames instead of queueing them up.
const SEND_EVERY_MS = 500
const SEND_WIDTH = 640          // plenty for a face or a coin, ~50 KB as JPEG
const JPEG_QUALITY = 0.7

export default function BrowserCamera({ baseUrl }) {
  const videoRef = useRef(null)
  // No mediaDevices at all means a plain-http address, where browsers withhold
  // the camera entirely - known before anything runs.
  const [error, setError] = useState(() => (navigator.mediaDevices
    ? '' : 'This page needs https (or localhost) to use the camera.'))

  useEffect(() => {
    if (!navigator.mediaDevices) return undefined
    let stream = null
    let timer = null
    let sending = false
    let stopped = false
    const canvas = document.createElement('canvas')

    const sendFrame = () => {
      const video = videoRef.current
      if (sending || !video || video.readyState < 2 || !video.videoWidth) return
      const scale = Math.min(1, SEND_WIDTH / video.videoWidth)
      canvas.width = Math.round(video.videoWidth * scale)
      canvas.height = Math.round(video.videoHeight * scale)
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
      sending = true
      canvas.toBlob((blob) => {
        if (!blob || stopped) { sending = false; return }
        const form = new FormData()
        form.append('file', blob, 'frame.jpg')
        fetch(`${baseUrl}/frame`, { method: 'POST', body: form })
          .catch(() => {})
          .finally(() => { sending = false })
      }, 'image/jpeg', JPEG_QUALITY)
    }

    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    })
      .then((s) => {
        if (stopped) { s.getTracks().forEach((t) => t.stop()); return }
        stream = s
        if (videoRef.current) videoRef.current.srcObject = s
        timer = setInterval(sendFrame, SEND_EVERY_MS)
      })
      .catch((e) => {
        setError(e?.name === 'NotAllowedError'
          ? 'Camera blocked — allow it in the address bar, then reload.'
          : 'No camera available in this browser.')
      })

    return () => {
      stopped = true
      clearInterval(timer)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [baseUrl])

  if (error) {
    return (
      <div className="flex h-full w-full items-center justify-center p-6 text-center text-sm text-white/60">
        {error}
      </div>
    )
  }
  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      aria-label="Live camera preview"
      className="h-full w-full object-contain"
    />
  )
}
