import { useState, useRef, useEffect, useCallback } from "react";
import { markSpeaking, hearingSelf, followAudio } from "./selfVoice";
// Her name lives in wake.js because the singing panel has to recognise it
// too — that is how the microphone gets handed back.
import { NAME, NAME_CJK } from "./wake";

// Uses the browser's built-in speech recognition (Chrome/Edge). This returns the
// user's ACTUAL words — or nothing on silence — so it never hallucinates random
// text the way audio-to-Gemini transcription does. It's also free (no quota).
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

// Recognition failures are invisible from the outside: a denied microphone, a
// network error and simply nobody speaking all look the same — nothing
// happens. Every one of those has been mistaken for "she is ignoring me", so
// the panel now records what it is doing where it can be read back.
const trace = (baseUrl, line) => {
  try {
    const fd = new FormData();
    // Tag the page. Two consoles on two ports each run their own recognition,
    // and the browser gives the microphone to one of them — without knowing
    // which page a line came from, that fight is invisible in the log.
    fd.append("line", "VOICE[" + location.port + "] " + line);
    fetch(`${baseUrl}/debug_log`, { method: "POST", body: fd }).catch(() => {});
  } catch { /* telemetry must never break the thing it watches */ }
};

// Greetings in CJK scripts (no \b word boundaries — they don't work for CJK).
const GREET_CJK =
  "(?:こんにちは|こんばんは|おはよう(?:ございます)?|やあ|ねえ|ハロー|ハイ|" +
  "안녕하세요|안녕|여보세요|" +
  "你好|您好|哈喽|哈囉|嗨|早上好|下午好|晚上好)";

// Wake phrase: her NAME, with or without a greeting in front of it.
//
// The greeting used to be required — "Hi Alzona" woke her, "Alzona" did not.
// That is not how anyone calls someone by name, and the telemetry showed it
// costing every attempt: she transcribed "Alzona. I.", "Elzona. Mama.",
// "Arizona, Arizona." perfectly and ignored all of them for want of a "hi".
// Calling a robot by her name IS the greeting.
//
// A bare name is deliberately enough to wake her but never enough to be taken
// as a command: with nothing after it she answers and waits, so the cost of a
// stray "Arizona" in conversation is one "Hello, I'm listening", not an action.
const WAKE_RE = new RegExp(
  "\\b(?:(?:hey|hi|hello|heya|yo|greetings|kumusta|kamusta|mabuhay|" +
  "good\\s+(?:morning|afternoon|evening|day)|" +
  "magandang\\s+(?:umaga|hapon|gabi|araw)|okay|ok)[ ,!.]*)?" + NAME + "\\b" +
  "|" + GREET_CJK + "[、。，,!！?？・\\s]*" + NAME_CJK +
  "|" + NAME_CJK,
  "i"
);

// Stop phrase: "Thank you, Alzona" (thanks/salamat/ありがとう/감사합니다/谢谢)
// -> back to standby.
const STOP_RE = new RegExp(
  "\\b(?:thank\\s*you|thanks|salamat)(?:\\s*po)?[ ,!.]*" + NAME + "\\b" +
  "|(?:ありがとう(?:ございます|ございました)?|どうも|" +
  "감사합니다|감사해요|고마워요?|고맙습니다|" +
  "谢谢|謝謝|多谢|多謝)[、。，,!！?？・\\s]*" + NAME_CJK,
  "i"
);

// Speech-recognition languages the user can pick from (the Web Speech API
// cannot auto-detect the spoken language — it needs to be told).
// Philippine English first, and the default.
//
// en-US was the default, and it is the wrong model for this room: it renders
// "Alzona" as "Alzana" and "Arizona", "Baybayin" as "be buying", and Filipino
// place and person names as whatever English words they resemble. en-PH is
// trained on exactly this accent and on Filipino proper nouns, and costs
// nothing to switch to — Chrome ships both.
// The five she is built for. Japanese and Korean were here and are gone: an
// option nobody selects still has to be right, and a language she is not
// prepared to answer well in is worse than one she does not offer.
//
// Dialects unchanged. en-PH stays the English default — it is the model
// trained on this accent, and en-US renders "Alzona" as "Alzana" and
// "Baybayin" as "be buying". es-ES for Spanish: Chrome has no Philippine
// Spanish, and es-ES is the closest to how it is taught here.
const SR_LANGS = [
  { code: "en-PH", label: "English (PH)" },
  { code: "en-US", label: "English (US)" },
  { code: "fil-PH", label: "Filipino" },
  { code: "es-ES", label: "Español" },
  { code: "hr-HR", label: "Hrvatski" },
  { code: "zh-CN", label: "中文" },
];

const DEFAULT_LANG = "en-PH";

// How sure the recogniser must be before she acts on what it returned.
// Chrome reports this per segment; where it reports nothing the transcript
// is taken at face value, since refusing everything would be worse than
// occasionally answering a bad guess.
const MIN_COMMAND_CONFIDENCE = 0.55;

export default function VoiceRecorder({
  baseUrl = "http://localhost:5002",
  onResult = () => {},
  // True while the singing panel holds the microphone. Recognition insists on
  // owning the device, so it must actually STOP — not merely ignore what it
  // hears — or it restarts the singing side out of the microphone every second.
  suspended = false,
  // Supplied by App so the whole console shares one identity. Anything the
  // backend answers later can then be matched to the console that asked.
  clientId: consoleId = "",
}) {

  const [recording, setRecording] = useState(false);
  const [responseText, setResponseText] = useState("");
  const [typedText, setTypedText] = useState("");
  const [sending, setSending] = useState(false);
  const [awake, setAwake] = useState(false);
  const [srLang, setSrLang] = useState(DEFAULT_LANG);
  // Set when the browser refuses the microphone for THIS origin. Worth its
  // own state because it is not a transient error: nothing will ever be
  // heard until someone grants it, and the panel otherwise looks merely idle.
  const [micBlocked, setMicBlocked] = useState(false);
  // True when ANOTHER console currently holds the microphone.
  const [micTaken, setMicTaken] = useState(false);
  // Something was heard clearly while she was asleep. Worth saying: from the
  // outside, ignoring a question and failing to hear it look identical.
  const [ignoredWhileAsleep, setIgnoredWhileAsleep] = useState("");

  // Ref mirrors so speech-recognition callbacks always see current values.
  const awakeRef = useRef(false);
  // Mirrored into a ref: the recognition callbacks and the 1s poll are both
  // closures created before a handover happens, and would otherwise keep
  // seeing the old value and grab the microphone straight back.
  const suspendedRef = useRef(false);
  const srLangRef = useRef(DEFAULT_LANG);
  const lastPollTrace = useRef(0);
  // True when ALZONA spoke at any point during the CURRENT recognition session.
  // hearingSelf() only guards the moment recognition starts; if she begins
  // talking while it is already running, the open session transcribes her and
  // hands her own answer back as though a visitor had said it.
  const spokeDuringRef = useRef(false);
  // The audio of the current utterance, kept so a bad reading can be checked
  // against Gemini. Null when recording is not possible on this machine.
  const clipRef = useRef(null);
  const clipStreamRef = useRef(null);
  // Set if recording ever costs us the recogniser. They are documented in
  // wake.js as fighting over the microphone on this hardware; if that happens
  // the recording is abandoned rather than breaking the thing it assists.
  const clipUnsafeRef = useRef(false);
  // Identifies this page to the lease. The port alone is not enough: two
  // tabs on the SAME console would then look like one holder to each other.
  const clientId = useRef(
    `${location.port || "80"}-${Math.random().toString(36).slice(2, 8)}`);

  const changeSrLang = (code) => {
    srLangRef.current = code;
    setSrLang(code);
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch { /* restarts with new lang */ }
    }
  };

  const wakeUp = () => {
    awakeRef.current = true;
    setAwake(true);
    setIgnoredWhileAsleep("");
  };

  const goToSleep = () => {
    awakeRef.current = false;
    setAwake(false);
  };

  const recognitionRef = useRef(null);
  // Blocks auto-listen while recording / processing / ALZONA is speaking, so the
  // mic can never pick up ALZONA's own voice.
  const busyRef = useRef(false);

  const finishBusy = () => {
    setTimeout(() => { busyRef.current = false; }, 800);    // cooldown after speaking
  };

  const sendText = async (text, alternatives = [], clip = null) => {
    try {
      const form = new FormData();
      form.append("text", text);
      if (consoleId) form.append("client", consoleId);
      // The audio, for the backend to fall back on when this text is not a
      // command it recognises. It decides; sending it costs nothing until then.
      if (clip) form.append("audio", clip, "utterance.webm");
      // Recognition returns several guesses and its favourite is not always the
      // right one: "translate pilipinas to baybayin" came back as "...to be
      // buying" with the correct reading further down the list. The backend
      // tries these when the top guess is not a command it knows, so a demo
      // does not hinge on the recogniser's first choice.
      if (alternatives.length) {
        form.append("alts", JSON.stringify(alternatives.slice(0, 5)));
      }
      form.append("skip_tts", "1");   // text now, audio in parallel via /say
      const res = await fetch(`${baseUrl}/command`, { method: "POST", body: form });
      const data = await res.json();
      // A 'youtube:<id>' marker is not a path on this server; prefixing it
      // with the backend URL produced a black frame and no error.
      const abs = (u) =>
        (!u ? null : u.startsWith("youtube:") ? u : `${baseUrl}${u}`);
      setResponseText(data.reply || text || "");
      onResult({
        transcript: data.transcript || text,
        reply: data.reply,
        mode: data.mode,
        image_url: abs(data.image_url),
        video_url: abs(data.video_url),
        command: data.command,
        // Forward these too. The backend answers a spoken or typed "harmonize
        // me in alto" with a sing directive and "identify this coin" with the
        // coin fields, and dropping them here meant the command was understood,
        // answered out loud, and then quietly had no effect on the panel it was
        // about. On the other console the singing panel has its own ear and
        // received them by another route, which is what hid this.
        sing: data.sing,
        coin: data.coin,
      });
      // Reply is already on screen; fetch and play the voice when it's ready.
      if (data.reply) {
        sayInAlzonaVoice(data.reply);
      } else {
        finishBusy();
      }
    } catch (err) {
      console.error("Command failed:", err);
      finishBusy();
    }
  };

  /** Ask the backend for the microphone. Renews the lease if we already hold it. */
  const claimMic = async () => {
    try {
      const fd = new FormData();
      fd.append("client", clientId.current);
      // A focused window outranks an unfocused one, so switching tabs moves the
      // microphone to the console being looked at.
      fd.append("focused", document.hasFocus() ? "1" : "0");
      const r = await fetch(`${baseUrl}/mic_lease`, { method: "POST", body: fd });
      if (!r.ok) return true;          // no arbiter reachable — carry on alone
      const d = await r.json();
      return !!d.yours;
    } catch {
      return true;                      // never let the lease be what breaks listening
    }
  };

  const releaseMic = () => {
    try {
      const fd = new FormData();
      fd.append("client", clientId.current);
      fd.append("release", "1");
      fetch(`${baseUrl}/mic_lease`, { method: "POST", body: fd }).catch(() => {});
    } catch { /* going away anyway */ }
  };

  /**
   * Record the same utterance the recogniser is hearing.
   *
   * Only ever used when the browser's reading turns out not to be a command —
   * see the backend — so the usual path costs nothing but the recording itself.
   *
   * Deliberately forgiving: on this hardware SpeechRecognition and a
   * getUserMedia capture have been seen to take the microphone from each other.
   * If that happens the recording is dropped for the rest of the session and
   * the recogniser is left alone. A worse transcript is survivable; losing the
   * microphone entirely is not.
   */
  const startClip = async () => {
    if (clipUnsafeRef.current || clipRef.current) return;
    if (typeof MediaRecorder === "undefined") return;
    try {
      const stream = clipStreamRef.current
        || await navigator.mediaDevices.getUserMedia({ audio: true });
      clipStreamRef.current = stream;
      const rec = new MediaRecorder(stream);
      const chunks = [];
      rec.ondataavailable = (e) => { if (e.data?.size) chunks.push(e.data); };
      rec.onerror = () => { clipUnsafeRef.current = true; };
      rec.start();
      clipRef.current = { rec, chunks };
    } catch (e) {
      clipUnsafeRef.current = true;
      trace(baseUrl, `clip recording unavailable (${e?.name || "error"}) — `
        + `browser transcript only`);
    }
  };

  /** The utterance just recorded, or null. Always stops the recorder. */
  const takeClip = async () => {
    const held = clipRef.current;
    clipRef.current = null;
    if (!held) return null;
    const { rec, chunks } = held;
    if (rec.state === "inactive") return null;
    await new Promise((done) => {
      rec.onstop = done;
      try { rec.stop(); } catch { done(); }
    });
    if (!chunks.length) return null;
    const blob = new Blob(chunks, { type: chunks[0].type || "audio/webm" });
    // Too short to carry a sentence; sending it would spend a Gemini call on
    // a click or a breath.
    return blob.size > 2000 ? blob : null;
  };

  const startRecording = () => {
    if (!SR || recording || busyRef.current || suspendedRef.current) {
      trace(baseUrl, `blocked sr=${!!SR} rec=${recording} `
        + `busy=${busyRef.current} suspended=${suspendedRef.current}`);
      return;
    }
    busyRef.current = true;
    const r = new SR();
    r.lang = srLangRef.current;   // user-selected recognition language
    r.interimResults = true;   // react to quiet speech as it forms
    r.maxAlternatives = 5;     // quiet audio often has the wake phrase in a lower-ranked guess
    r.continuous = true;       // don't cut off at the first brief pause

    spokeDuringRef.current = false;   // fresh session, she has not spoken in it
    let finals = [];           // best transcript of each finished segment
    let alts = [];             // every alternative heard (checked for wake/stop phrases)
    let confs = [];            // how sure the recogniser was about each segment
    let stopTimer = null;

    // Stop ~1.2s after the last recognition activity, so slow or soft
    // speakers aren't cut off mid-sentence.
    const scheduleStop = () => {
      clearTimeout(stopTimer);
      stopTimer = setTimeout(() => { try { r.stop(); } catch { /* already stopped */ } }, 1200);
    };

    r.onstart = () => {
      setRecording(true);
      setMicBlocked(false);
      trace(baseUrl, 'listening');
      startClip();
    };
    r.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const seg = e.results[i];
        if (!seg.isFinal) continue;
        const best = (seg[0].transcript || "").trim();
        if (best) {
          finals.push(best);
          // Chrome reports 0 for confidence on some builds; only real numbers
          // are worth keeping, or every segment would look hopeless.
          if (typeof seg[0].confidence === "number" && seg[0].confidence > 0) {
            confs.push(seg[0].confidence);
          }
        }
        for (let j = 0; j < seg.length; j++) {
          const t = (seg[j].transcript || "").trim();
          if (t) alts.push(t);
        }
      }
      scheduleStop();          // any activity (even interim) extends the window
    };
    // "no-speech" is ordinary and handled in onend; "not-allowed" means the
    // microphone was refused and nothing will ever work until that is fixed.
    r.onerror = (e) => {
      const err = e?.error || 'unknown';
      trace(baseUrl, `error ${err}`);
      // Permission is granted per ORIGIN — scheme, host AND port — so allowing
      // the microphone on one console grants nothing to the other, which runs
      // on a different port. That trips people up every time.
      if (err === 'not-allowed' || err === 'service-not-allowed') setMicBlocked(true);
    };
    r.onend = () => {
      clearTimeout(stopTimer);
      setRecording(false);

      if (spokeDuringRef.current) {
        // She talked over this session. Whatever was captured is her own reply
        // coming back through the speakers — acting on it starts a loop where
        // she answers herself.
        trace(baseUrl, "discarded — ALZONA was speaking during this session");
        finishBusy();
        return;
      }

      const heard = finals.join(" ").trim();
      // Best transcript first, then recognition alternatives as fallbacks.
      const candidates = heard ? [heard, ...alts] : alts;
      const confidence = confs.length
        ? confs.reduce((a, b) => a + b, 0) / confs.length
        : null;
      trace(baseUrl, `heard=${JSON.stringify(heard)} `
        + `alts=${alts.length} conf=${confidence === null ? '?' : confidence.toFixed(2)} `
        + `lang=${srLangRef.current} awake=${awakeRef.current}`);
      if (!candidates.length) {
        takeClip();            // stop the recorder; nothing to check
        finishBusy();          // silence -> do nothing (no hallucination)
        return;
      }

      // "Thank you, Alzona" -> stop accepting voice input.
      if (awakeRef.current && candidates.some((t) => STOP_RE.test(t))) {
        goToSleep();
        sayInAlzonaVoice("You're welcome! Just greet me again when you need me.");
        return;
      }

      // Find the wake phrase in the best transcript OR any alternative.
      let wakeSource = null;
      let wakeMatch = null;
      for (const t of candidates) {
        const m = t.match(WAKE_RE);
        if (m) { wakeSource = t; wakeMatch = m; break; }
      }

      if (!awakeRef.current) {
        if (!wakeMatch) {
          // Asleep and not addressed. She does nothing — but a visitor who
          // just asked a full question deserves to know why nothing happened.
          // Measured live: "What is the most famous festival in Croatia?" was
          // transcribed perfectly three times and dropped three times, with
          // the panel showing only "Microphone ready".
          const words = heard.trim().split(/\s+/).filter(Boolean);
          if (words.length >= 3) {
            setIgnoredWhileAsleep(heard.trim());
            trace(baseUrl, `asleep — ignored ${JSON.stringify(heard)}`);
          }
          finishBusy();
          return;
        }
        // Greeting + name heard -> ALZONA is awake until the stop phrase.
        wakeUp();
        const command = wakeSource
          .slice(wakeMatch.index + wakeMatch[0].length)
          .replace(/^[\s,.!?]+/, "")
          .trim();
        if (command) {
          takeClip().then((clip) => sendText(command, candidates, clip));
        } else {
          sayInAlzonaVoice("Hello! I'm listening. How can I help you?");
        }
        return;
      }

      // Awake, but the recogniser is guessing.
      //
      // Once awake, every transcript is treated as something to answer — so a
      // bad guess is not merely ignored, it is answered out loud, and her reply
      // keeps her talking while the next bad guess arrives. Live, that produced
      // "Hello Bucking.", "Send the mic, mum." and "Alzheimer's." in a row, each
      // dutifully replied to.
      //
      // Below this the audio did not carry. Saying nothing is the honest
      // response to something that was not understood; a wake phrase still gets
      // through above, because recognising her name is a lower bar than
      // transcribing a sentence.
      if (confidence !== null && confidence < MIN_COMMAND_CONFIDENCE) {
        trace(baseUrl, `ignored — only ${confidence.toFixed(2)} sure of `
          + `${JSON.stringify(heard)}`);
        finishBusy();
        return;
      }

      // Awake: everything is a command (strip a repeated greeting if present).
      let command = heard || candidates[0];
      if (wakeMatch && wakeSource === command) {
        const rest = command
          .slice(wakeMatch.index + wakeMatch[0].length)
          .replace(/^[\s,.!?]+/, "")
          .trim();
        if (rest) command = rest;
      }
      takeClip().then((clip) => sendText(command, candidates, clip));
    };

    recognitionRef.current = r;
    try { r.start(); } catch { busyRef.current = false; }
  };

  const stopRecording = useCallback(() => {
    if (recognitionRef.current) recognitionRef.current.stop();
  }, []);

  // Hand the microphone over to the singing panel. Nothing here waits for
  // recognition to finish its sentence: the singer is already singing, and a
  // recogniser holding the device is exactly what stops the harmony hearing
  // them. Coming back the other way needs no action — the 1s poll re-arms on
  // its own once `suspended` clears.
  useEffect(() => {
    suspendedRef.current = suspended;
    if (suspended) {
      try { stopRecording(); } catch { /* already stopped */ }
    }
    // stopRecording is redefined every render; the body is idempotent, so
    // re-running it costs nothing and keeps the dependency list honest.
  }, [suspended, stopRecording]);

  // Speak a short phrase in ALZONA's own voice (backend /say). Falls back to
  // the browser voice only if the backend can't synthesize (e.g. TTS quota).
  const sayInAlzonaVoice = async (text) => {
    // Anything the microphone picks up from here until this recognition
    // session ends is hers, not a visitor's.
    spokeDuringRef.current = true;

    // Close the microphone before saying anything.
    //
    // It used to stay open through her whole reply and simply discard whatever
    // it heard. Discarding was enough to stop her answering herself, but the
    // session was still running: a visitor who spoke while she was talking had
    // their words captured into a session that was going to be thrown away, so
    // they went unanswered and nobody could tell why. Closing it means she is
    // plainly not listening while she speaks, and the next thing said is heard
    // from the start.
    //
    // busyRef is already true here and is cleared by finishBusy once the audio
    // ends, so the poll that restarts listening waits for her — see
    // startRecording, which refuses while busy.
    const listening = recognitionRef.current;
    if (listening) {
      recognitionRef.current = null;
      try {
        // abort, not stop: stop delivers whatever it has and fires onend with
        // results, which is a turn nobody took.
        if (listening.abort) listening.abort();
        else listening.stop();
      } catch { /* already gone */ }
    }

    // Open the speaking window NOW, before the audio exists.
    //
    // She is about to talk, and everything that waits for her to finish asks
    // this window. It used to open only once /say had answered — so for the
    // length of that fetch she was "not speaking", and the harmony took that
    // as its cue: the count-in started, and the reply then played over it.
    // Held open on a timer because the fetch has no fixed duration, and
    // released the moment real playback takes over the marking.
    const holdWhileFetching = setInterval(() => markSpeaking(1200), 300);
    markSpeaking(1200);
    const doneFetching = () => clearInterval(holdWhileFetching);

    try {
      const form = new FormData();
      form.append("text", text);
      // No "voice" field: the backend's default order, ElevenLabs first
      // (0.8-1.2s measured) and Gemini only if ElevenLabs fails (4.7-5.3s).
      // Asking for "gemini" here made every reply wait on the slower voice.
      const res = await fetch(`${baseUrl}/say`, { method: "POST", body: form });
      const data = await res.json();
      if (data.tts_url) {
        // followAudio, not just a flag here: this plays out of the same
        // speakers the singing panel's microphone is listening to, and that
        // panel has no other way of knowing this reply is hers.
        const audio = followAudio(new Audio(`${baseUrl}${data.tts_url}?t=${Date.now()}`));
        audio.onended = () => { doneFetching(); finishBusy(); };
        audio.onerror = () => { doneFetching(); finishBusy(); };
        markSpeaking();          // cover the gap before the first timeupdate
        await audio.play();
        doneFetching();          // timeupdate keeps the window open from here
        return;
      }
    } catch {
      // fall through to browser voice
    }
    // Every path from here has its own way of holding the window open, so the
    // fetch timer stops. Left running it would keep her "speaking" for ever and
    // nothing waiting on her would ever start.
    doneFetching();
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(text);
      // Synthesis reports no progress events worth trusting, so hold the
      // window open on a timer and let it lapse the moment speaking stops.
      const holdOpen = setInterval(() => markSpeaking(), 250);

      // And give up holding it after however long this could possibly take.
      //
      // onend was the ONLY thing clearing that interval. A tab that will not
      // let audio play, or a machine with no installed voice, never fires it —
      // and then she counts as speaking for ever. Measured: the dance clip,
      // which waits for her to stop, started at 15.25s, exactly its own
      // give-up cap, on every single attempt. Everything that waits on her
      // voice was waiting on a timer that had already failed.
      //
      // Speech runs at roughly fifteen characters a second; three seconds of
      // margin covers a slow voice and a slow start.
      const longestItCouldTake = (text.length / 15) * 1000 + 3000;
      const giveUp = setTimeout(() => {
        clearInterval(holdOpen);
        finishBusy();
      }, longestItCouldTake);

      const done = () => {
        clearInterval(holdOpen);
        clearTimeout(giveUp);
        markSpeaking();
        finishBusy();
      };
      u.onend = done;
      u.onerror = done;
      window.speechSynthesis.cancel();
      markSpeaking();
      window.speechSynthesis.speak(u);
    } else {
      finishBusy();
    }
  };

  // Typed input goes through the same /command path as voice.
  const handleTypedSubmit = async (e) => {
    e.preventDefault();
    const text = typedText.trim();
    if (!text || sending) return;
    busyRef.current = true;      // pause auto-listen while ALZONA responds
    setSending(true);
    setTypedText("");
    await sendText(text);
    setSending(false);
  };

  // Auto-listen when a face is present, but only when not busy.
  // Release on the way out. Releasing after every TURN instead made the two
  // consoles alternate: one finished, the other grabbed it mid-restart, and
  // both kept aborting. Whoever is being spoken to should simply keep it.
  useEffect(() => () => releaseMic(), []);

  useEffect(() => {
    if (suspendedRef.current) releaseMic();   // the singing panel has the mic now
  }, [suspended]);

  // Take the microphone the moment this window is focused, instead of waiting
  // for the next poll — clicking a console should make it the one listening.
  useEffect(() => {
    const grab = () => { if (!suspendedRef.current) claimMic(); };
    window.addEventListener("focus", grab);
    document.addEventListener("visibilitychange", grab);
    return () => {
      window.removeEventListener("focus", grab);
      document.removeEventListener("visibilitychange", grab);
    };
  }, []);

  useEffect(() => {
    if (!SR) return;
    const interval = setInterval(async () => {
      try {
        // Renew while actually listening. The readiness test below requires
        // NOT recording, so without this the holder stopped renewing exactly
        // while it was using the microphone, the lease lapsed, and the other
        // console took it mid-sentence.
        if (recording || busyRef.current) { claimMic(); }

        const res = await fetch(`${baseUrl}/state`);
        if (!res.ok) return;
        const data = await res.json();
        // Deliberately NOT gated on the camera seeing a face.
        //
        // It used to be. The cost was a robot that went silent for reasons
        // nobody could see: pointed at a wall, a visitor standing slightly off
        // to one side, poor light — she simply stopped answering, and every
        // time it read as a broken microphone. The face was never what made
        // listening safe anyway; her NAME is. Nothing is acted on until she
        // hears it, so the camera has no say in whether she can hear at all.
        // A hidden tab has no business holding the microphone: nobody is
        // talking to a console they cannot see, and holding it there is what
        // made the visible one look deaf.
        if (document.visibilityState === "hidden") {
          if (!recording) releaseMic();
          return;
        }

        const ready =
          !recording &&
          !busyRef.current &&
          !suspendedRef.current &&
          !hearingSelf(performance.now());
        if (ready) {
          // Only one console may listen at a time. Without this both start,
          // the browser aborts one, and they take turns failing in silence.
          const mine = await claimMic();
          setMicTaken(!mine);
          if (mine) startRecording();
          else if (performance.now() - lastPollTrace.current > 5000) {
            lastPollTrace.current = performance.now();
            trace(baseUrl, 'standing down — another console holds the microphone');
          }
        } else if (performance.now() - lastPollTrace.current > 5000) {
          // Throttled: one line every 5s is enough to see which gate is shut,
          // without burying the singing telemetry in the same log.
          lastPollTrace.current = performance.now();
          trace(baseUrl, `idle face=${data.face_state} (not a gate) rec=${recording} `
            + `busy=${busyRef.current} suspended=${suspendedRef.current} `
            + `self=${hearingSelf(performance.now())}`);
        }
      } catch {
        // silent
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [recording, baseUrl]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        {awake ? (
          <span className="text-sm font-semibold text-emerald-400">
            Awake — listening (say “Thank you, Alzona” to stop)
          </span>
        ) : recording ? (
          <span className="text-sm text-amber-400">
            Standby — say “Alzona” to wake her
          </span>
        ) : micTaken ? (
          <span className="text-sm font-semibold text-amber-300">
            Another ALZONA console is using the microphone — close that tab to
            talk to this one.
          </span>
        ) : micBlocked ? (
          <span className="text-sm font-semibold text-rose-300">
            Microphone blocked for this page — allow it from the icon in the
            address bar. Permission is per port, so allowing it on another
            console does not cover this one.
          </span>
        ) : (
          <span className="text-sm text-white/60">Microphone ready</span>
        )}
        {!SR && <span className="text-sm text-red-400">Use Chrome/Edge for voice</span>}
        <select
          value={srLang}
          onChange={(e) => changeSrLang(e.target.value)}
          title="Microphone language"
          className="ml-auto rounded-lg border border-white/15 bg-black/30 px-2 py-1 text-xs text-white/85 outline-none focus:border-sky-400/60"
        >
          {SR_LANGS.map((l) => (
            <option key={l.code} value={l.code} className="bg-[#171457]">
              {l.label}
            </option>
          ))}
        </select>
      </div>

      {!awake && ignoredWhileAsleep && (
        <p className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-xs leading-relaxed text-amber-200">
          Heard “{ignoredWhileAsleep}” — but she is in standby. Say{' '}
          <b>“Alzona”</b> first, or put her name in the question.
        </p>
      )}

      <form onSubmit={handleTypedSubmit} className="flex items-center gap-2">
        <input
          type="text"
          value={typedText}
          onChange={(e) => setTypedText(e.target.value)}
          placeholder="Type a message..."
          className="min-w-0 flex-1 rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-sm text-white placeholder-white/40 outline-none focus:border-sky-400/60"
        />
        <button
          type="submit"
          disabled={sending || !typedText.trim()}
          className="rounded-lg border border-sky-400/30 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-200 transition hover:bg-sky-400/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {sending ? "..." : "Send"}
        </button>
      </form>

      {/* <div className="mt-2 rounded-lg bg-[rgba(0,0,0,0.18)] p-3">
        <p className="text-xs uppercase tracking-wider text-white/60">AI Response</p>
        <p className="mt-1 text-sm text-white/90">{responseText || 'No transcript yet.'}</p>
      </div> */}
    </div>
  );
}
