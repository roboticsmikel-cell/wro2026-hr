import { useEffect, useRef, useState } from 'react'
import novusLogo from './assets/Novus.png'
import WRO26Logo from './assets/WRO26.png'
// Using Tailwind CSS for styling; remove App.css to avoid conflicts
import VoiceRecorder from './Voice'
import Sing from './Sing'
import Coin from './Coin'
import { markSpeaking, whenQuiet } from './selfVoice'

const initialState = {
  face_state: 'idle',
  age_result: null,
  chat: false,
  running: false,
}

function StateItem({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.05)] px-4 py-2">
      <p className="text-xs uppercase tracking-[0.35em] text-white/50">
        {label}
      </p>
      <p className="mt-2 text-lg font-semibold text-white">
        {String(value ?? '—')}
      </p>
    </div>
  )
}

/**
 * The ALZONA console.
 *
 * `SingPanel` is the one thing that varies between the two consoles. They are
 * the same program otherwise — same camera, chat, coin and Baybayin — and that
 * is deliberate: keeping them as one App with two entry points means a fix to
 * any shared feature lands in both, which copying the file would not give.
 *
 *   Sing    she listens, finds your place and follows you
 *   Leader  she sounds the note, counts in, and starts
 */
function App({ SingPanel = Sing }) {
  // main.py serves the backend here. Use localhost on the same PC, or your PC's
  // LAN IP (e.g. http://192.168.1.85:5002) when opening the app from another device.
  // Whatever host served this page is the host running the backend. Hardcoding
  // localhost worked only on this machine: opened from a phone, "localhost"
  // means the PHONE, so every request died and the site looked broken.
  const BASE_URL = `http://${window.location.hostname}:5002`
  const [backendState, setBackendState] = useState(initialState)
  const [transcript, setTranscript] = useState('')
  const [reply, setReply] = useState('')
  const [audioSrc, setAudioSrc] = useState(null)
  const [imageSrc, setImageSrc] = useState(null)   // Baybayin image
  const [videoSrc, setVideoSrc] = useState(null)   // folk-dance / teaching video
  // Where the clip starts and how long it runs. A visitor watches a clip,
  // not a performance, and the next person should not be waiting through it.
  const [videoClip, setVideoClip] = useState({ start: 0, seconds: 0 })
  const [mode, setMode] = useState('chat')
  const [knowledgeFiles, setKnowledgeFiles] = useState([])
  const [uploadingKnowledge, setUploadingKnowledge] = useState(false)
  const [coinFields, setCoinFields] = useState(null)  // the coin rows to show
  const [coinVerdict, setCoinVerdict] = useState(null)  // real / fake / unclear
  // The last coin answer already shown, so a background read is spoken once.
  const lastCoinSeq = useRef(0)
  // Cancels a clip still waiting for her to stop talking. Without it, asking
  // about a second dance leaves the first one queued behind the answer.
  const cancelPendingVideo = useRef(null)
  // This console's identity, for anything the backend answers later.
  //
  // The coin is read in the background and delivered on /state, which EVERY
  // console polls — so all of them saw the answer and all of them spoke it.
  // With three consoles open that is three voices reading the same sentences
  // over each other. The answer now carries who asked, and only they say it.
  const consoleId = useRef(
    `${location.port || '80'}-${Math.random().toString(36).slice(2, 8)}`)
  const [singCommand, setSingCommand] = useState(null) // armed by a spoken command
  // Who holds the microphone. Exactly one of the two panels may: the speech
  // recogniser in Voice owns the device while it runs, so the singing panel
  // cannot listen at the same time — they simply restart each other. False
  // means the conversation has it, which is the resting state.
  const [singing, setSinging] = useState(false)


  const audioRef = useRef(null)

  const refreshKnowledge = async () => {
    try {
      const res = await fetch(`${BASE_URL}/knowledge`)
      if (!res.ok) return
      const data = await res.json()
      setKnowledgeFiles(data.files || [])
    } catch {
      // backend offline — leave the list as is
    }
  }

  // Refetch when the backend comes online (running flips false -> true),
  // so the list isn't stuck empty if the page loaded before the backend.
  useEffect(() => {
    refreshKnowledge()
  }, [backendState.running])

  const handleKnowledgeUpload = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    setUploadingKnowledge(true)
    try {
      const res = await fetch(`${BASE_URL}/upload_knowledge`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (data?.error) {
        console.error('Knowledge upload error:', data.error)
        alert(data.error)
      }
      await refreshKnowledge()
    } catch (error) {
      console.error('Knowledge upload failed:', error)
    } finally {
      setUploadingKnowledge(false)
    }
  }

  const handleKnowledgeDelete = async (name) => {
    try {
      await fetch(`${BASE_URL}/knowledge/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })
      await refreshKnowledge()
    } catch (error) {
      console.error('Knowledge delete failed:', error)
    }
  }

  // Update visuals only. Audio playback (and its timing) is handled in Voice.jsx
  // so the mic can't capture ALZONA's own voice.
  const applyResult = (res) => {
    setTranscript(res.transcript || '')
    setReply(res.reply || '')
    setMode(res.mode || 'chat')
    setImageSrc(res.image_url ? res.image_url + `?t=${Date.now()}` : null)

    // The clip waits for her to finish saying what it is.
    //
    // Starting it as soon as the answer arrives put the dance music straight
    // over the sentence describing the dance — two sounds at once, and the
    // sentence is the part nobody can replay. The picture appears with the
    // words; only the sound has to wait.
    setVideoClip({ start: res.video_start || 0, seconds: res.video_seconds || 0 })
    if (cancelPendingVideo.current) cancelPendingVideo.current()
    if (res.video_url) {
      cancelPendingVideo.current = whenQuiet(() => {
        cancelPendingVideo.current = null
        setVideoSrc(res.video_url)
      })
    } else {
      setVideoSrc(null)
    }
    // A spoken "identify this coin" fills the coin panel just like the button.
    if (res.coin) setCoinFields(res.coin)
    // A spoken "harmonize with me in alto" arms the singing panel AND hands it
    // the microphone. Arming alone used to be enough only because the singing
    // panel was always listening anyway.
    if (res.sing) {
      setSingCommand(res.sing)
      setSinging(true)
    }
  }

  useEffect(() => {
    let isMounted = true

    const syncState = async () => {
      try {
        const response = await fetch(`${BASE_URL}/state`)

        if (!response.ok) return

        const data = await response.json()

        if (isMounted) {
          setBackendState(data)

          // A coin she was ASKED about, read in the background while she was
          // already talking. The sequence number is what distinguishes a new
          // answer from the one already on screen — comparing the fields would
          // miss someone holding up the same coin twice.
          const coin = data.coin
          if (coin && coin.seq && coin.seq !== lastCoinSeq.current) {
            lastCoinSeq.current = coin.seq
            // Someone else's question. The panel is shared, so the fields are
            // still worth showing, but the answer is not ours to say.
            const mine = !coin.asked_by || coin.asked_by === consoleId.current
            if (coin.fields) {
              setCoinFields(coin.fields)
              setCoinVerdict(coin.verdict ?? null)
            }
            const say = mine ? (coin.spoken || coin.error) : ''
            if (say) {
              setReply(say)
              // Her own voice, through the same route as any other reply, so
              // the singing panel still knows the sound is hers.
              const form = new FormData()
              form.append('text', say)
              // No 'voice' field: ElevenLabs first, Gemini only if it fails.
              fetch(`${BASE_URL}/say`, { method: 'POST', body: form })
                .then((r) => r.json())
                .then((d) => {
                  if (d.tts_url) setAudioSrc(`${BASE_URL}${d.tts_url}?t=${Date.now()}`)
                })
                .catch(() => {})
            }
          }
        }
      } catch (error) {
        console.error('State sync failed:', error)
      }
    }

    syncState()

    const intervalId = window.setInterval(syncState, 1000)

    return () => {
      isMounted = false
      window.clearInterval(intervalId)
    }
  }, [])

  useEffect(() => {
    if (!audioRef.current || !audioSrc) return

    const audio = audioRef.current

    audio.load()

    audio.play().catch((err) => {
      console.error('Autoplay blocked:', err)
    })
  }, [audioSrc])

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]

    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${BASE_URL}/upload_audio`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (data?.error) {
        console.error('Upload error:', data.error)
        return
      }

      // A 'youtube:<id>' marker is not a path on this server; prefixing it
      // with the backend URL produced 'http://host:5002youtube:ID' and a
      // black frame with no error.
      const abs = (u) =>
        (!u ? null : u.startsWith('youtube:') ? u : `${BASE_URL}${u}`)
      applyResult({
        transcript: data.transcript,
        reply: data.reply,
        mode: data.mode,
        tts_url: abs(data.tts_url),
        image_url: abs(data.image_url),
        video_url: abs(data.video_url),
      })
    } catch (error) {
      console.error('Upload failed:', error)
    }
  }

  return (
    <div className="min-h-screen bg-[#171457] px-6 py-2 text-white">
      <div className="mx-auto flex w-full max-w-7xl flex-col items-center gap-8">

        {/* Hidden Audio Player */}
        <audio
          ref={audioRef}
          autoPlay
          hidden
          src={audioSrc || undefined}
          onPlay={() => markSpeaking()}
          onPlaying={() => markSpeaking()}
          onTimeUpdate={() => markSpeaking()}
          onEnded={() => markSpeaking()}
          onPause={() => markSpeaking()}
        />

        {/* Header */}
        <div className="flex flex-row items-center gap-6 text-center lg:gap-8">
          <img
            src={novusLogo}
            alt="Novus logo"
            className="h-32 w-32 lg:h-44 lg:w-44"
          />

          <div className="max-w-3xl">
            <p className="font-extrabold text-6xl tracking-[0.18em] text-[#ffe8b6] lg:text-7xl">
              ALZONA
            </p>

            <p className="mt-3 text-lg font-semibold text-[#ffe8b6] lg:text-xl">
              <b>A</b>ndroid for <b>L</b>earners as <b>Z</b>one and 
            </p>

            <p className="text-lg font-semibold text-[#ffe8b6] lg:text-xl">
              <b>O</b>asis of <b>N</b>ational <b>A</b>rchives
            </p>
          </div>

          <img
            src={WRO26Logo}
            alt="WRO26 logo"
            className="h-32 w-32 lg:h-44 lg:w-44"
          />
        </div>

        {/* Main Grid */}
        <div className="grid w-full gap-6 lg:grid-cols-3">

          {/* Column 1: Frame Preview */}
          <section className="lg:col-span-2 rounded-3xl border border-white/10 bg-[rgba(255,255,255,0.06)] p-6 shadow-2xl shadow-black/20 backdrop-blur-sm flex flex-col">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.35em] text-white/50">
                  Camera
                </p>

                <h2 className="mt-2 text-3xl font-bold text-white">
                  Frame Preview
                </h2>
              </div>

              <div className="rounded-full border  border-sky-400/30 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-200">
                /video
              </div>
            </div>

            <div className="relative mt-6 overflow-hidden rounded-2xl h-64 w-full max-w-full lg:h-full md:h-420px border border-white/10 bg-black">
              {/* Live camera (always mounted so the stream stays warm) */}
              <img
                src={`${BASE_URL}/video`}
                alt="Live camera preview"
                className="h-full w-full object-contain"
              />

              {/* Baybayin image overlays the camera frame */}
              {imageSrc && (
                <img
                  src={imageSrc}
                  alt="Baybayin"
                  className="absolute inset-0 h-full w-full bg-white object-contain p-4"
                />
              )}

              {/* Video presentation plays inside the camera frame */}
              {videoSrc && (videoSrc.startsWith('youtube:') ? (
                /* A YouTube clip. A plain <video> cannot play one — it wants a
                   media file, and given a YouTube page it shows a black frame
                   and no error, which looks exactly like a broken feature. */
                // start/end are the player's own parameters, so YouTube stops
                // itself — more reliable than a timer racing a buffering video.
                <iframe
                  title="Dance"
                  src={`https://www.youtube.com/embed/${videoSrc.slice(8)}`
                    + `?autoplay=1&rel=0&start=${videoClip.start}`
                    + (videoClip.seconds
                        ? `&end=${videoClip.start + videoClip.seconds}` : '')}
                  allow="accelerometer; autoplay; encrypted-media; picture-in-picture"
                  allowFullScreen
                  className="absolute inset-0 h-full w-full border-0 bg-black"
                />
              ) : (
                <video
                  src={videoSrc}
                  autoPlay
                  controls
                  onLoadedMetadata={(e) => {
                    if (videoClip.start) e.currentTarget.currentTime = videoClip.start
                  }}
                  onTimeUpdate={(e) => {
                    // The file has no idea it is being clipped, so the page
                    // has to stop it. Checked on timeupdate rather than a
                    // timer: a video that buffers would otherwise be cut off
                    // having played less than the thirty seconds promised.
                    if (!videoClip.seconds) return
                    const done = videoClip.start + videoClip.seconds
                    if (e.currentTarget.currentTime >= done) {
                      e.currentTarget.pause()
                      setVideoSrc(null)
                    }
                  }}
                  onEnded={() => setVideoSrc(null)}
                  className="absolute inset-0 h-full w-full bg-black object-contain"
                />
              ))}
            </div>
          </section>

          {/* Column 2: Speech I/O */}

          <section className="lg:col-span-1 rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.06)] p-4 py-6shadow-2xl shadow-black/20 backdrop-blur-sm">
            <div className="flex items-center justify-between gap-4 mt-4">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-white/50">
                  Backend
                </p>

                <h2 className="mt-1 text-lg font-bold text-white">
                  Live State
                </h2>
              </div>

              <div className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                /state
              </div>
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <StateItem
                label="face_state"
                value={backendState.face_state}
              />
              <StateItem
                label="age_result"
                value={backendState.age_result}
              />
              <StateItem
                label="chat"
                value={backendState.chat}
              />
              <StateItem
                label="running"
                value={backendState.running}
              />
            </div>
            <p className="text-xs pt-4 uppercase tracking-[0.35em] text-white/50">
              Speech
            </p>

            <h2 className="mt-1 text-2xl font-bold text-white">
              I/O
            </h2>

            <div className="mt-3 space-y-2">

              <div>
                <VoiceRecorder baseUrl={BASE_URL} onResult={applyResult}
                  suspended={singing} clientId={consoleId.current} />
              </div>

              <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.35em] text-white/45">
                  Transcript
                </p>

                <p className="mt-2 min-h-8 max-h-32 overflow-y-auto whitespace-pre-wrap break-words text-sm text-white/90">
                  {transcript || 'No transcript yet.'}
                </p>
              </div>

              <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.35em] text-white/45">
                  Reply {mode !== 'chat' ? `· ${mode}` : ''}
                </p>

                <p className="mt-2 min-h-8 max-h-64 overflow-y-auto whitespace-pre-wrap break-words text-sm text-white/90">
                  {reply || 'No reply yet.'}
                </p>
              </div>

              {/* Baybayin & videos now display inside the live-camera frame.
                  The sheet is also auto-sent to the printer — surface how that
                  went so a paper jam isn't silent. */}
              {backendState.print_status?.word && (
                <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                  <p className="text-xs uppercase tracking-[0.35em] text-white/45">
                    Printer
                  </p>
                  <p
                    className={`mt-2 text-sm ${
                      backendState.print_status.ok ? 'text-emerald-300' : 'text-rose-300'
                    }`}
                  >
                    {backendState.print_status.ok
                      ? `Printed "${backendState.print_status.word}" → ${backendState.print_status.detail}`
                      : `Print failed: ${backendState.print_status.detail}`}
                  </p>
                </div>
              )}

              <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs uppercase tracking-[0.35em] text-white/45">
                    Knowledge
                  </p>
                  <label className="cursor-pointer rounded-lg border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-xs font-semibold text-sky-200 transition hover:bg-sky-400/20">
                    {uploadingKnowledge ? 'Uploading...' : '+ Add file'}
                    <input
                      type="file"
                      accept=".txt,.md,.csv,.pdf,.docx,.jsonl,.json"
                      onChange={handleKnowledgeUpload}
                      disabled={uploadingKnowledge}
                      hidden
                    />
                  </label>
                </div>

                {knowledgeFiles.length === 0 ? (
                  <p className="mt-2 text-sm text-white/50">
                    No files yet. Upload .txt, .md, .csv, .pdf, .docx, .jsonl
                    or .json and ALZONA will search them when answering.
                  </p>
                ) : (
                  <ul className="mt-2 max-h-32 space-y-1 overflow-y-auto">
                    {knowledgeFiles.map((f) => (
                      <li
                        key={f.name}
                        className="flex items-center justify-between gap-2 text-sm text-white/85"
                      >
                        <span className="min-w-0 truncate">{f.name}</span>
                        <button
                          onClick={() => handleKnowledgeDelete(f.name)}
                          className="shrink-0 text-xs text-red-300/80 hover:text-red-300"
                          title="Remove from knowledge library"
                        >
                          remove
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

            </div>
          </section>
        </div>

        {/* Row 2: singing + coin identification */}
        <div className="grid w-full gap-6 lg:grid-cols-2">
          <SingPanel
            baseUrl={BASE_URL}
            armed={singCommand}
            active={singing}
            // Set from three places: the spoken trigger above, "Alzona" heard
            // while singing, and the button on the panel itself.
            onActiveChange={(on) => {
              setSinging(on)
              if (!on) setSingCommand(null)
            }}
            onClear={() => setSingCommand(null)}
            // ALZONA hears everything through the singing panel's microphone.
            // When what she heard was a question rather than singing, the answer
            // arrives here and is shown and spoken exactly like a typed one.
            onHeardSpeech={(data) => {
              applyResult({
                transcript: data.transcript,
                reply: data.reply,
                mode: data.mode,
                image_url: data.image_url ? `${BASE_URL}${data.image_url}` : null,
                video_url: !data.video_url ? null
                  : data.video_url.startsWith('youtube:') ? data.video_url
                  : `${BASE_URL}${data.video_url}`,
                coin: data.coin,
                sing: data.sing,
              })
              if (data.tts_url) setAudioSrc(`${BASE_URL}${data.tts_url}?t=${Date.now()}`)
            }}
          />
          <Coin
            baseUrl={BASE_URL}
            fields={coinFields}
            verdict={coinVerdict}
            onResult={(fields, ttsUrl, verdict) => {
              setCoinFields(fields)
              setCoinVerdict(verdict ?? null)
              if (ttsUrl) setAudioSrc(ttsUrl)
            }}
          />
        </div>
      </div>
    </div>
  )
}

export default App



