"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Mic,
  MicOff,
  ShieldAlert,
  Shield,
  ShieldCheck,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Radio,
  Clock,
  Zap,
  User,
  X,
} from "lucide-react";
import { SpectrogramPanel } from "@/components/SpectrogramPanel";
import { Navbar } from "@/components/Navbar";

// ── Types ─────────────────────────────────────────────────────────────────────
interface VoiceResult {
  score: number;
  verdict: string;
  severity: string;
  reasons: string[];
  spectrogram_image: string | null;
  raw?: {
    transcript?: string;
    acoustic_score?: number;
    nlp_score?: number;
    deepfake_score?: number;
    is_deepfake?: boolean;
  };
}

interface LiveChunk {
  chunk_count: number;
  current_score: number;
  verdict: string;
  elapsed_seconds: number;
  high_risk_triggered: boolean;
  time_to_alert_seconds?: number;
  spectrogram_image: string | null;
  reasons: string[];
  nlp_intent?: string;
}

interface ChatMessage {
  speaker: "You" | "Caller" | "Unknown";
  text: string;
  timestamp: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_BASE  = API_BASE.replace(/^http/, "ws");

function parseChunkTranscript(chunkText: string): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  
  if (!chunkText) return [];
  
  const regex = /\[(You|Person \d+)\]:\s*/g;
  const matches = [...chunkText.matchAll(regex)];
  
  if (matches.length === 0) {
    messages.push({
      speaker: "Caller",
      text: chunkText.trim(),
      timestamp: timeStr
    });
  } else {
    for (let i = 0; i < matches.length; i++) {
      const match = matches[i];
      const speakerRaw = match[1];
      const speaker = speakerRaw === "You" ? "You" : "Caller";
      const startIdx = match.index! + match[0].length;
      const endIdx = i + 1 < matches.length ? matches[i + 1].index! : chunkText.length;
      const text = chunkText.substring(startIdx, endIdx).trim();
      
      if (text) {
        messages.push({
          speaker,
          text,
          timestamp: timeStr
        });
      }
    }
  }
  return messages;
}

function highlightKeywords(text: string) {
  const keywords = [
    // Banking & credential theft
    "fraud", "fraud message", "otp", "one time password", "cvv", "card number",
    "account number", "net banking", "atm", "account blocked", "unauthorized transaction",
    "debit card", "credit card", "ifsc",
    // Identity documents
    "aadhaar", "aadhar", "pan card", "pan number", "passport", "voter id",
    "kyc", "kyc expired", "link aadhaar", "mandatory verification",
    // Remote access & tech support
    "anydesk", "teamviewer", "remote access", "screen share", "download app",
    "install software", "ip address", "hacked", "virus",
    // Digital Arrest & law enforcement impersonation
    "arrest", "digital arrest", "police", "cbi", "customs", "court order",
    "warrant", "judiciary", "enforcement directorate", "money laundering",
    "criminal case", "fir", "cyber cell", "rbi",
    // UPI & payments fraud
    "upi pin", "gpay", "phonepe", "scan qr", "qr code", "paytm", "bhim",
    "collect request", "cashback", "approved transfer",
    // Lottery & prize scams
    "lottery", "prize money", "lucky draw", "winner", "claim prize",
    "registration fee", "you have won", "gift card", "free prize",
    // Emergency & family scams
    "accident", "hospital deposit", "kidnapped", "emergency transfer",
    "bail money", "stuck abroad", "send immediately",
    // Loan scams
    "loan approved", "pre-approved", "processing fee", "emi waiver",
    "instant loan", "zero interest",
    // Coercion markers
    "immediately", "urgently", "right now", "do not tell anyone",
    "keep this secret", "last warning",
    // Hindi keywords
    "giraftari", "paisa", "paise", "khata", "incident", "jail",
    "ओटीपी", "आधार", "पुलिस", "पैसे", "गिरफ्तारी", "अकाउंट", "ब्लॉक"
  ];
  
  if (!text) return "";
  
  const escapedKeywords = keywords.map(k => k.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
  const regex = new RegExp(`\\b(${escapedKeywords.join('|')})\\b|(${keywords.filter(k => /[^\x00-\x7F]/.test(k)).join('|')})`, 'gi');
  
  const parts = text.split(regex);
  return (
    <span>
      {parts.map((part, i) => {
        if (!part) return null;
        const isKeyword = keywords.some(k => k.toLowerCase() === part.toLowerCase());
        return isKeyword ? (
          <span key={i} className="text-red-400 font-bold bg-red-500/10 px-1 rounded shadow-sm border border-red-500/20 animate-pulse">
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </span>
  );
}

interface AdvisoryInfo {
  status: "SAFE" | "SUSPICIOUS" | "WARNING" | "CRITICAL";
  color: string;
  bgColor: string;
  borderColor: string;
  icon: React.ReactNode;
  heading: string;
  description: string;
  actions: string[];
}

function getAdvisory(
  score: number,
  intent: string,
  ShieldAlertIcon: React.ReactNode,
  AlertTriangleIcon: React.ReactNode,
  ShieldIcon: React.ReactNode
): AdvisoryInfo {
  if (score >= 75) {
    const scamType = intent ? intent.replace(/_/g, ' ').toUpperCase() : "FRAUD SCAM";
    return {
      status: "CRITICAL",
      color: "text-red-500",
      bgColor: "bg-red-500/5",
      borderColor: "border-red-500/30",
      icon: ShieldAlertIcon,
      heading: `CRITICAL DETECTED: ${scamType}`,
      description: "High-probability scam call patterns matching known fraud tactics. The caller is attempting to obtain sensitive credentials or device access.",
      actions: [
        "🔴 HANG UP IMMEDIATELY. Do not continue talking.",
        "🔒 Do NOT disclose any OTP, passwords, UPI PINs, or bank card details.",
        "🖥️ Never download software like Anydesk or TeamViewer if requested.",
        "📞 Block this sender/caller number on WhatsApp and Report Fraud."
      ]
    };
  }
  
  if (score >= 50) {
    return {
      status: "WARNING",
      color: "text-amber-500",
      bgColor: "bg-amber-500/5",
      borderColor: "border-amber-500/30",
      icon: AlertTriangleIcon,
      heading: "MODERATE RISK: SUSPICIOUS ACTIVITY",
      description: "Call center background noise matched or pressure tactics are detected in speech patterns. Intent is likely social engineering.",
      actions: [
        "⚠️ Avoid answering personal or family status questions.",
        "🔍 Verify credentials of caller through official channels directly.",
        "⛔ Do not transfer any money, advance fees, or process quick deposits.",
        "⏸️ Tell the caller you will call them back and check their info."
      ]
    };
  }
  
  if (score >= 25) {
    return {
      status: "SUSPICIOUS",
      color: "text-yellow-400",
      bgColor: "bg-yellow-400/5",
      borderColor: "border-yellow-400/20",
      icon: AlertTriangleIcon,
      heading: "ELEVATED CONCERN",
      description: "Minor anomalies or common phrasing cues detected. Stay alert to any sudden escalation of terms.",
      actions: [
        "👀 Listen carefully to what credentials or links are requested.",
        "💳 Keep details secure. No legal entity calls out of the blue asking to verify accounts."
      ]
    };
  }
  
  return {
    status: "SAFE",
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/5",
    borderColor: "border-emerald-500/20",
    icon: ShieldIcon,
    heading: "SYSTEM ACTIVE: STATUS SECURE",
    description: "Ongoing call transcription shows low risk signature. Regular conversation pattern detected.",
    actions: [
      "🟢 Normal conversation. If they ask for remote sharing or bank links, the score will update instantly.",
      "🛡️ End-to-end local transcription and acoustic analysis is protecting this call."
    ]
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function scoreColor(score: number) {
  if (score >= 75) return "text-red-500";
  if (score >= 50) return "text-yellow-400";
  if (score >= 25) return "text-amber-500";
  return "text-emerald-500";
}
function scoreBorder(score: number) {
  if (score >= 75) return "border-red-500";
  if (score >= 50) return "border-yellow-500";
  if (score >= 25) return "border-amber-500";
  return "border-emerald-500";
}
function scoreRing(score: number) {
  if (score >= 75) return "stroke-red-500";
  if (score >= 50) return "stroke-yellow-500";
  if (score >= 25) return "stroke-amber-500";
  return "stroke-emerald-500";
}
function scoreBandLabel(score: number, raw?: any): string {
  if (score >= 75) {
    let category = "";
    if (raw?.nlp_intent) {
      category = raw.nlp_intent;
    } else if (raw?.raw?.nlp_intent) {
      category = raw.raw.nlp_intent;
    }
    if (category && category !== 'unknown' && category !== 'legitimate') {
      return category.replace(/_/g, ' ').toUpperCase();
    }
    return "FRAUD";
  }
  if (score >= 50) return "ACCEPTABLE";
  if (score >= 25) return "SUSPICIOUS";
  return "SAFE";
}
function scoreBandSubtext(score: number): string {
  if (score >= 75) return "High risk — fraud/scam detected";
  if (score >= 50) return "Acceptable — some minor flags";
  if (score >= 25) return "Caution — suspicious activity";
  return "Safe — no threats found";
}

// ── WAV Encoding Helpers ──────────────────────────────────────────────────────
function encodeWAV(buffers: Float32Array[], sampleRate: number, numChannels: number = 1): ArrayBuffer {
  const bytesPerSample = 2;
  const numSamples = buffers.reduce((acc, b) => acc + b.length, 0);
  const buffer = new ArrayBuffer(44 + numSamples * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + numSamples * bytesPerSample, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, numSamples * bytesPerSample, true);

  let offset = 44;
  for (const b of buffers) {
    for (let i = 0; i < b.length; i++) {
      let s = Math.max(-1, Math.min(1, b[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
  }

  return buffer;
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// ── Sub-components ────────────────────────────────────────────────────────────
const ScoreDial = ({ score }: { score: number }) => (
  <div className="relative w-36 h-36 flex items-center justify-center">
    <svg className="w-full h-full -rotate-90" viewBox="0 0 144 144">
      <circle cx="72" cy="72" r="60" className="stroke-surface-high fill-none" strokeWidth="8" />
      <motion.circle
        initial={{ strokeDasharray: "0, 1000" }}
        animate={{ strokeDasharray: `${(score / 100) * 377}, 1000` }}
        transition={{ duration: 1.2, ease: "easeOut" }}
        cx="72" cy="72" r="60"
        className={`fill-none ${scoreRing(score)}`}
        strokeWidth="8"
        strokeLinecap="round"
      />
    </svg>
    <div className="absolute inset-0 flex flex-col items-center justify-center">
      <span className={`text-4xl font-headline font-bold ${scoreColor(score)} glow-text`}>{Number(score).toFixed(1)}</span>
      <span className="text-[8px] font-headline font-bold text-on-surface-variant tracking-widest">RISK SCORE</span>
    </div>
  </div>
);

const ReasonList = ({ reasons }: { reasons: string[] }) => (
  <ul className="space-y-1.5 max-h-40 overflow-y-auto pr-1 scrollbar-thin">
    {reasons.slice(0, 8).map((r, i) => (
      <li key={i} className="flex items-start gap-2 text-[10px] text-on-surface-variant font-light leading-relaxed">
        <Zap className="w-2.5 h-2.5 mt-0.5 text-primary/60 shrink-0" />
        {r}
      </li>
    ))}
  </ul>
);

const getCountryDetails = (num: string) => {
  const clean = num.replace(/\s+/g, "").replace(/-/g, "");
  if (clean.startsWith("+91")) return { code: "+91", name: "India", flag: "🇮🇳" };
  if (clean.startsWith("+1")) return { code: "+1", name: "United States / Canada", flag: "🇺🇸" };
  if (clean.startsWith("+44")) return { code: "+44", name: "United Kingdom", flag: "🇬🇧" };
  if (clean.startsWith("+61")) return { code: "+61", name: "Australia", flag: "🇦🇺" };
  if (clean.startsWith("+81")) return { code: "+81", name: "Japan", flag: "🇯🇵" };
  if (clean.startsWith("+49")) return { code: "+49", name: "Germany", flag: "🇩🇪" };
  if (clean.startsWith("+33")) return { code: "+33", name: "France", flag: "🇫🇷" };
  if (clean.startsWith("+86")) return { code: "+86", name: "China", flag: "🇨🇳" };
  if (clean.startsWith("+39")) return { code: "+39", name: "Italy", flag: "🇮🇹" };
  if (clean.startsWith("+7")) return { code: "+7", name: "Russia", flag: "🇷🇺" };
  if (clean.startsWith("+55")) return { code: "+55", name: "Brazil", flag: "🇧🇷" };
  
  const m = clean.match(/^\+(\d{1,4})/);
  if (m) {
    return { code: `+${m[1]}`, name: "International Address", flag: "🌐" };
  }
  return { code: "Unknown", name: "Unknown / Spoofed Origin", flag: "❓" };
};

const formatDuration = (sec: number): string => {
  const hrs = Math.floor(sec / 3600);
  const mins = Math.floor((sec % 3600) / 60);
  const secs = sec % 60;
  return [
    hrs > 0 ? String(hrs).padStart(2, '0') : null,
    String(mins).padStart(2, '0'),
    String(secs).padStart(2, '0')
  ].filter((v): v is string => v !== null).join(':');
};

// normalise a phone string for fuzzy matching
const normPhone = (p: string) => p.replace(/[\s\-()]+/g, "");

export default function VoiceAnalyzerPage() {
  // ── Upload tab state ───────────────────────────────────────────────────────
  const [file, setFile]         = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<VoiceResult | null>(null);
  const [uploadError, setUploadError]   = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Contacts state ─────────────────────────────────────────────────────────
  type Contact = { name: string; phone: string };
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [contactsLoaded, setContactsLoaded] = useState(false);
  const [contactError, setContactError] = useState<string | null>(null);
  const [contactSuccess, setContactSuccess] = useState<string | null>(null);
  const [newContactName, setNewContactName] = useState("");
  const [newContactPhone, setNewContactPhone] = useState("");
  const [savingContact, setSavingContact] = useState(false);
  const [showContactManager, setShowContactManager] = useState(false);

  // Load contacts from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/contacts`)
      .then(r => r.json())
      .then(d => { setContacts(d.contacts ?? []); setContactsLoaded(true); })
      .catch(() => { setContactsLoaded(true); });
  }, []);

  // runtime lookup against the live contacts list
  const findSavedContact = (num: string): Contact | undefined => {
    const clean = normPhone(num);
    if (!clean) return undefined;
    return contacts.find(c => {
      const cleanC = normPhone(c.phone);
      return clean === cleanC
        || (clean.length >= 7 && cleanC.endsWith(clean))
        || (cleanC.length >= 7 && clean.endsWith(cleanC));
    });
  };

  const showMsg = (type: "ok" | "err", msg: string) => {
    if (type === "ok") { setContactSuccess(msg); setContactError(null); }
    else { setContactError(msg); setContactSuccess(null); }
    setTimeout(() => { setContactSuccess(null); setContactError(null); }, 3500);
  };

  const addContact = async () => {
    const name = newContactName.trim();
    const phone = newContactPhone.trim();
    if (!name || !phone) return;

    // Optimistic update — show immediately in UI
    const optimistic: Contact = { name, phone };
    setContacts(prev => [
      ...prev.filter(c => normPhone(c.phone) !== normPhone(phone)),
      optimistic
    ]);
    setNewContactName("");
    setNewContactPhone("");
    setSavingContact(true);

    try {
      const fd = new FormData();
      fd.append("name", name);
      fd.append("phone", phone);
      const res = await fetch(`${API_BASE}/contacts`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      // Sync with server truth
      if (Array.isArray(data.contacts)) setContacts(data.contacts);
      showMsg("ok", `${name} saved to whitelist`);
    } catch (err: any) {
      showMsg("err", `Could not reach backend: ${err.message ?? err}. Contact saved locally only.`);
    } finally {
      setSavingContact(false);
    }
  };

  const removeContact = async (phone: string) => {
    // Optimistic remove
    setContacts(prev => prev.filter(c => normPhone(c.phone) !== normPhone(phone)));
    try {
      const res = await fetch(`${API_BASE}/contacts?phone=${encodeURIComponent(phone)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data.contacts)) setContacts(data.contacts);
    } catch (err: any) {
      showMsg("err", `Remove failed: ${err.message ?? err}`);
    }
  };

  // ── Live call tab state ────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"upload" | "live">("upload");
  const [callId, setCallId]       = useState(() => `call-${Date.now().toString(36)}`);
  const [wsStatus, setWsStatus]   = useState<"idle" | "connecting" | "connected" | "ended">("idle");
  const [chunks, setChunks]       = useState<LiveChunk[]>([]);
  const [latestSpec, setLatestSpec] = useState<string | null>(null);
  const [latestSpecChunk, setLatestSpecChunk] = useState<number | undefined>();
  const wsRef = useRef<WebSocket | null>(null);
  const [liveMessages, setLiveMessages] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [callerNumber, setCallerNumber] = useState("");
  const [liveDuration, setLiveDuration] = useState(0);
  const [manualText, setManualText] = useState("");
  const [manualSpeaker, setManualSpeaker] = useState<"You" | "Caller">("Caller");
  const [injectingText, setInjectingText] = useState(false);
  const [contactName, setContactName] = useState<string | null>(null);
  // Browser-side real-time transcript (SpeechRecognition API — zero latency)
  const speechRecRef = useRef<any>(null);
  const [liveInterim, setLiveInterim] = useState(""); // current partial sentence

  // Start/stop browser Web Speech API for instant live transcript
  const startSpeechRecognition = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return; // Safari/Firefox fallback — backend transcript only
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-IN";
    rec.onresult = (event: any) => {
      let interim = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalText += t;
        } else {
          interim += t;
        }
      }
      setLiveInterim(interim);
      if (finalText.trim()) {
        const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        setLiveMessages(prev => [...prev, { speaker: "You", text: finalText.trim(), timestamp: timeStr }]);
        setLiveInterim("");
      }
    };
    rec.onerror = () => {};
    rec.onend = () => {
      // Auto-restart so it doesn't stop after 60s silence
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try { rec.start(); } catch (_) {}
      }
    };
    try { rec.start(); } catch (_) {}
    speechRecRef.current = rec;
  };

  const stopSpeechRecognition = () => {
    try { speechRecRef.current?.stop(); } catch (_) {}
    speechRecRef.current = null;
    setLiveInterim("");
  };


  useEffect(() => {
    let timer: any = null;
    if (wsStatus === "connected") {
      timer = setInterval(() => {
        setLiveDuration(prev => prev + 1);
      }, 1000);
    } else {
      setLiveDuration(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [wsStatus]);


  // ── Browser Audio Capture states ──────────────────────────────────────────
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamsRef = useRef<MediaStream[]>([]);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const sampleBufferRef = useRef<Float32Array[]>([]);
  const recordIntervalRef = useRef<any>(null);

  // Autoscroll live calls chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [liveMessages]);

  // ── Drag-drop ──────────────────────────────────────────────────────────────
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  // ── Upload & analyze ───────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    setUploadError(null);

    try {
      const fd = new FormData();
      fd.append("audio", file);
      const res = await fetch(`${API_BASE}/analyze/voice`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      let displayVerdict = data.verdict ?? "UNKNOWN";
      const raw = data.raw;
      let category = "";
      if (raw?.nlp_intent) {
        category = raw.nlp_intent;
      } else if (raw?.raw?.nlp_intent) {
        category = raw.raw.nlp_intent;
      }
      
      const uploadScore = Math.round(data.score ?? 100);
      const isThreat = uploadScore <= 50;
      if (isThreat && category && category !== 'unknown' && category !== 'legitimate') {
        displayVerdict = category.replace(/_/g, ' ').toUpperCase();
      }

      setUploadResult({
        score:    uploadScore,
        verdict:  displayVerdict,
        severity: data.severity ?? "NONE",
        reasons:  data.reasons ?? [],
        spectrogram_image: data.spectrogram_image ?? null,
        raw: data.raw,
      });
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setUploading(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // LIVE CALL TAB
  // ─────────────────────────────────────────────────────────────────────────────
  const [fullTranscript, setFullTranscript] = useState("");

  const connectWs = (
    captureAudio: boolean = false,
    micStream: MediaStream | null = null,
    tabStream: MediaStream | null = null
  ) => {
    const id = `call-${Date.now().toString(36)}`;
    setCallId(id);
    setChunks([]);
    setFullTranscript("");
    setLiveMessages([]);
    setLatestSpec(null);
    setWsStatus("connecting");
    setCaptureError(null);
    setContactName(null);

    // Look up if caller number matches a saved contact
    const matched = findSavedContact(callerNumber);
    const actualCapture = matched ? false : captureAudio;
    setIsCapturing(actualCapture);

    const ws = new WebSocket(`${WS_BASE}/ws/production-live-call/${id}?caller_number=${encodeURIComponent(callerNumber)}`);
    wsRef.current = ws;

    ws.onopen = async () => {
      setWsStatus("connected");
      startSpeechRecognition(); // browser real-time transcript — zero latency
      if (actualCapture) {
        // Streams already obtained from user-gesture click — pass them directly
        await startAudioCapture(ws, micStream, tabStream);
      }
    };
    
    ws.onerror = () => {
      setWsStatus("ended");
      stopSpeechRecognition();
      cleanupCapture();
    };
    
    ws.onclose = () => {
      setWsStatus("ended");
      stopSpeechRecognition();
      cleanupCapture();
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);

        if (msg.type === "contact_verified") {
          setContactName(msg.name);
          const chunk: LiveChunk = {
            chunk_count: 1,
            current_score: 0,
            verdict: "SAFE",
            elapsed_seconds: 0,
            high_risk_triggered: false,
            spectrogram_image: null,
            reasons: [`Identity verified with protected Whitelist: ${msg.name}. Security clearance active.`],
            nlp_intent: "legitimate",
          };
          setChunks([chunk]);
          return;
        }

        // ── Fast transcript packet — appears before ML score ──────────────
        if (msg.type === "transcript_ready") {
          const segmentText = msg.transcript_segment;
          if (segmentText?.trim()) {
            setFullTranscript(prev => (prev + " " + segmentText).trim());
            const parsed = parseChunkTranscript(segmentText);
            if (parsed.length > 0) {
              setLiveMessages(prev => [...prev, ...parsed]);
            }
          }
          return;
        }

        if (msg.type === "chunk_result") {
          const chunkScore = msg.threat_score ?? msg.current_score ?? 0;
          const chunkReasons = msg.explainable_reasons ?? msg.reasons ?? [];
          const hasAlertTriggered = msg.alerts_triggered ?? msg.high_risk_triggered ?? false;
          
          const chunk: LiveChunk = {
            chunk_count:           msg.chunk_count ?? chunks.length + 1,
            current_score:         chunkScore,
            verdict:               msg.verdict ?? "SAFE",
            elapsed_seconds:       msg.timestamp ?? msg.elapsed_seconds ?? 0,
            high_risk_triggered:   hasAlertTriggered,
            time_to_alert_seconds: msg.time_to_alert_seconds,
            spectrogram_image:     msg.spectrogram_image ?? null,
            reasons:               chunkReasons,
            nlp_intent:            msg.nlp_intent ?? msg.verdict,
          };
          setChunks((prev) => [...prev.slice(-19), chunk]);
          
          if (msg.is_saved_contact && msg.contact_name) {
            setContactName(msg.contact_name);
          }

          // Only add transcript text if it wasn't already shown by transcript_ready
          const segmentText = msg.transcript_segment ?? msg.current_chunk_transcript;
          if (segmentText?.trim()) {
            setFullTranscript(prev => {
              // Avoid duplicating if transcript_ready already added it
              if (prev.includes(segmentText.trim())) return prev;
              return (prev + " " + segmentText).trim();
            });
            // Don't add to liveMessages again — transcript_ready already did
          }

          if (chunk.spectrogram_image) {
            setLatestSpec(chunk.spectrogram_image);
            setLatestSpecChunk(chunk.chunk_count);
          }
        }
      } catch (err) {
        console.error("WS message parse error:", err);
      }
    };
  };

  const injectManualTranscript = async (e: React.FormEvent) => {
    e.preventDefault();
    const textToInject = manualText.trim();
    if (!textToInject) return;

    setInjectingText(true);
    try {
      const fd = new FormData();
      fd.append("message", textToInject);
      fd.append("sender", manualSpeaker === "Caller" ? callerNumber : "client");
      fd.append("channel", "voice");

      const res = await fetch(`${API_BASE}/analyze/text`, {
        method: "POST",
        body: fd
      });

      if (!res.ok) throw new Error("Classifier request failed");
      const data = await res.json();
      const combined = data.combined ?? data;

      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setLiveMessages(prev => [...prev, {
        speaker: manualSpeaker === "You" ? "You" : "Caller",
        text: textToInject,
        timestamp: timeStr
      }]);

      const chunkScore = combined.score ?? 0;
      const chunkVerdict = combined.verdict ?? "SAFE";
      const chunkReasons = combined.reasons ?? [];
      const nlpIntent = data.components?.text?.raw?.nlp_intent ?? data.raw?.nlp_intent ?? "unknown";

      const simulatedChunk: LiveChunk = {
        chunk_count: chunks.length + 1,
        current_score: chunkScore,
        verdict: chunkVerdict,
        elapsed_seconds: liveDuration,
        high_risk_triggered: chunkScore >= 70,
        spectrogram_image: null,
        reasons: chunkReasons.length > 0 ? chunkReasons : ["Manual transcript verification scan completed."],
        nlp_intent: nlpIntent
      };

      setChunks(prev => [...prev.slice(-19), simulatedChunk]);
      setManualText("");
    } catch (err) {
      console.error("Manual injection failed:", err);
    } finally {
      setInjectingText(false);
    }
  };

  // Called from user-gesture onClick — acquires media permissions directly, then connects
  const handleCaptureConnect = async () => {
    setCaptureError(null);
    let micStream: MediaStream | null = null;
    let tabStream: MediaStream | null = null;

    // Step 1: Mic — must be direct user gesture
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (e) {
      console.warn("Mic denied:", e);
    }

    // Step 2: Tab / screen share — MUST be from user gesture (click handler)
    // Chrome blocks this if called from a network callback like ws.onopen
    try {
      tabStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "browser" } as any,
        audio: true,
      });
    } catch (e: any) {
      // User cancelled or denied — soft warning, not a hard failure
      if (micStream) {
        setCaptureError(
          "Tab sharing was cancelled — capturing mic only. " +
          "For full call analysis, click Connect again, choose the WhatsApp Web tab, and tick \"Share tab audio\"."
        );
      } else {
        setCaptureError(
          "No audio permissions granted. Session will run in manual-text mode. " +
          "Use the 'Inject Statement' box below."
        );
      }
    }

    // Step 3: Warn if tab was granted but has no audio track
    if (tabStream) {
      const audioTracks = tabStream.getAudioTracks();
      if (audioTracks.length === 0) {
        setCaptureError(
          "Tab shared but without audio. Re-connect and tick \"Share tab audio\" in the browser dialog."
        );
        // Stop the video-only stream; treat as no tab
        tabStream.getTracks().forEach(t => t.stop());
        tabStream = null;
      }
    }

    // Step 4: Now open the WebSocket and pass the pre-obtained streams
    connectWs(true, micStream, tabStream);
  };

  // startAudioCapture now receives pre-obtained streams (no more getDisplayMedia inside)
  const startAudioCapture = async (
    ws: WebSocket,
    micStream: MediaStream | null,
    tabStream: MediaStream | null
  ) => {
    // Register streams for cleanup
    if (micStream) mediaStreamsRef.current.push(micStream);
    if (tabStream) mediaStreamsRef.current.push(tabStream);

    const tabAudioTracks = tabStream ? tabStream.getAudioTracks() : [];
    const hasMic = micStream && micStream.getAudioTracks().length > 0;
    const hasTab = tabAudioTracks.length > 0;

    if (!hasMic && !hasTab) {
      setIsCapturing(false);
      return; // Capture error already set by handleCaptureConnect
    }

    setIsCapturing(true);

    // Build AudioContext at 16 kHz
    const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioCtx = new AudioCtxClass({ sampleRate: 16000 });
    audioContextRef.current = audioCtx;

    const bufferSize = 4096;
    const processor = audioCtx.createScriptProcessor(bufferSize, 2, 2);
    scriptProcessorRef.current = processor;

    const merger = audioCtx.createChannelMerger(2);

    // Left channel = mic (your voice)
    if (hasMic) {
      const micSource = audioCtx.createMediaStreamSource(micStream!);
      micSource.connect(merger, 0, 0);
    }

    // Right channel = tab audio (caller's voice)
    if (hasTab) {
      const tabSource = audioCtx.createMediaStreamSource(new MediaStream(tabAudioTracks));
      tabSource.connect(merger, 0, 1);
    }

    merger.connect(processor);

    const silenceGain = audioCtx.createGain();
    silenceGain.gain.value = 0;
    processor.connect(silenceGain);
    silenceGain.connect(audioCtx.destination);

    sampleBufferRef.current = [];
    processor.onaudioprocess = (event) => {
      const left  = event.inputBuffer.getChannelData(0);
      const right = event.inputBuffer.getChannelData(1);
      const interleaved = new Float32Array(left.length + right.length);
      for (let i = 0; i < left.length; i++) {
        interleaved[i * 2]     = left[i];
        interleaved[i * 2 + 1] = right[i];
      }
      sampleBufferRef.current.push(interleaved);
    };

    // Every 1 s: encode as 16-bit 16 kHz stereo WAV and stream to backend
    recordIntervalRef.current = setInterval(() => {
      const buffers = sampleBufferRef.current;
      if (buffers.length === 0) return;
      sampleBufferRef.current = [];
      const wavBuffer = encodeWAV(buffers, 16000, 2);
      if (ws.readyState === WebSocket.OPEN) ws.send(wavBuffer);
    }, 1000);
  };

  const cleanupCapture = () => {
    if (recordIntervalRef.current) {
      clearInterval(recordIntervalRef.current);
      recordIntervalRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    mediaStreamsRef.current.forEach(stream => {
      stream.getTracks().forEach(track => track.stop());
    });
    mediaStreamsRef.current = [];
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    sampleBufferRef.current = [];
    setIsCapturing(false);
  };

  const disconnectWs = () => {
    try { wsRef.current?.send("END"); } catch (_) {}
    try { wsRef.current?.close(); } catch (_) {}
    wsRef.current = null;
    setWsStatus("ended");
    setContactName(null);
    stopSpeechRecognition();
    cleanupCapture();
  };

  // cleanup on unmount
  useEffect(() => () => {
    wsRef.current?.close();
    if (recordIntervalRef.current) clearInterval(recordIntervalRef.current);
    mediaStreamsRef.current.forEach(stream => {
      stream.getTracks().forEach(track => track.stop());
    });
  }, []);

  const latestChunk = chunks[chunks.length - 1] || (
    (wsStatus === "connected" || wsStatus === "connecting") ? {
      chunk_count: 0,
      current_score: 0,
      verdict: "SAFE",
      elapsed_seconds: liveDuration,
      high_risk_triggered: false,
      spectrogram_image: null,
      reasons: ["Awaiting voice activity or manual transcript injects..."],
      nlp_intent: "unknown"
    } as LiveChunk : undefined
  );

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen bg-background text-on-background pt-32 pb-16 px-6 relative">
      <Navbar />
      <div className="max-w-5xl mx-auto space-y-10">

        {/* ── Page header ─────────────────────────────────────────────── */}
        <div className="text-center space-y-3">
          <p className="text-primary font-headline font-bold uppercase tracking-[0.3em] text-[10px]">
            Spectral Forensics
          </p>
          <h1 className="text-4xl md:text-5xl font-headline font-bold text-white tracking-tight">
            Voice <span className="text-primary italic">Deepfake</span> Detector
          </h1>
          <p className="text-on-surface-variant font-light text-base max-w-xl mx-auto">
            Upload an audio file or connect a live WebSocket call to run acoustic
            + spectral + NLP analysis and see the mel-spectrogram fingerprint.
          </p>
        </div>

        {/* ── Tab switcher ────────────────────────────────────────────── */}
        <div className="flex justify-center gap-2">
          {(["upload", "live"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-2.5 rounded-xl text-xs font-headline font-bold uppercase tracking-widest transition-all ${
                activeTab === tab
                  ? "bg-primary text-black shadow-lg shadow-primary/20"
                  : "border border-outline/20 text-on-surface-variant hover:border-primary/30 hover:text-white"
              }`}
            >
              {tab === "upload" ? "Upload File" : "Live Call"}
            </button>
          ))}
        </div>

        {/* ─────────────────────────────────────────────────────────────── */}
        {/* UPLOAD TAB                                                      */}
        {/* ─────────────────────────────────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {activeTab === "upload" && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-6"
            >
              {/* Drop zone */}
              <div
                onDrop={onDrop}
                onDragOver={(e) => e.preventDefault()}
                className="glass-panel rounded-2xl border-2 border-dashed border-outline/20 hover:border-primary/30 transition-colors p-10 text-center cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
                />
                <Upload className="w-10 h-10 text-primary/40 mx-auto mb-3" />
                {file ? (
                  <p className="text-white font-medium">{file.name}</p>
                ) : (
                  <p className="text-on-surface-variant font-light text-sm">
                    Drop a WAV / MP3 / OGG file here, or click to browse
                  </p>
                )}
              </div>

              <div className="flex justify-center">
                <motion.button
                  disabled={!file || uploading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleAnalyze}
                  className="bg-primary text-black font-headline font-bold px-12 py-4 rounded-2xl shadow-lg shadow-primary/20 hover:bg-primary-dark transition-all disabled:opacity-40 flex items-center gap-3"
                >
                  {uploading ? <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing…</> : <><Shield className="w-5 h-5" /> Run Analysis</>}
                </motion.button>
              </div>

              {uploadError && (
                <p className="text-center text-error text-sm">{uploadError}</p>
              )}

              {/* Results */}
              <AnimatePresence>
                {uploadResult && (
                  <motion.div
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="grid md:grid-cols-2 gap-6"
                  >
                    {/* Left: score + verdict */}
                    <div className={`glass-panel rounded-2xl border-l-4 ${scoreBorder(uploadResult.score)} p-6 space-y-5`}>
                      <div className="flex items-center gap-4">
                        <ScoreDial score={uploadResult.score} />
                        <div>
                          <p className={`text-2xl font-headline font-bold ${scoreColor(uploadResult.score)}`}>
                            {uploadResult.verdict}
                          </p>
                          <div className="mt-1">
                            <span className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-headline font-bold border ${scoreBorder(uploadResult.score)} ${scoreColor(uploadResult.score)} bg-surface-high/30`}>
                              {scoreBandLabel(uploadResult.score, uploadResult.raw)}
                            </span>
                            <p className="text-[10px] text-on-surface-variant/80 font-light mt-0.5">
                              {scoreBandSubtext(uploadResult.score)}
                            </p>
                          </div>
                          <p className="text-xs text-on-surface-variant">{uploadResult.severity} severity</p>
                          {uploadResult.raw?.is_deepfake && (
                            <span className="mt-2 inline-block text-[9px] font-bold bg-error/10 text-error border border-error/30 px-2 py-0.5 rounded-full uppercase tracking-widest">
                              Deepfake Detected
                            </span>
                          )}
                        </div>
                      </div>

                      {uploadResult.raw && (
                        <div className="grid grid-cols-3 gap-2 text-center">
                          {[
                            { label: "Acoustic", val: uploadResult.raw.acoustic_score },
                            { label: "NLP",      val: uploadResult.raw.nlp_score },
                            { label: "Deepfake", val: uploadResult.raw.deepfake_score },
                          ].map(({ label, val }) => (
                            <div key={label} className="bg-surface/40 rounded-xl p-2">
                              <p className="text-[9px] text-on-surface-variant uppercase tracking-widest">{label}</p>
                              <p className="text-base font-headline font-bold text-white">{val?.toFixed(1) ?? "–"}</p>
                            </div>
                          ))}
                        </div>
                      )}

                      {uploadResult.raw?.transcript && (
                        <div className="bg-surface/30 rounded-xl p-3">
                          <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-1">Transcript</p>
                          <p className="text-[11px] text-on-surface-variant font-light leading-relaxed line-clamp-3">
                            {uploadResult.raw.transcript}
                          </p>
                        </div>
                      )}

                      <div>
                        <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-2">Detection Reasons</p>
                        <ReasonList reasons={uploadResult.reasons} />
                      </div>
                    </div>

                    {/* Right: spectrogram */}
                    <SpectrogramPanel
                      src={uploadResult.spectrogram_image}
                      label="Audio Spectrogram Analysis"
                      live={false}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {/* ─────────────────────────────────────────────────────────────── */}
          {/* LIVE CALL TAB                                                   */}
          {/* ─────────────────────────────────────────────────────────────── */}
          {activeTab === "live" && (
            <motion.div
              key="live"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-6"
            >
              {/* Connection controls */}
              <div className="glass-panel rounded-2xl p-6 space-y-4">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6 flex-1">
                    <div className="space-y-1">
                      <p className="text-xs text-on-surface-variant font-light">WebSocket Session ID</p>
                      <p className="font-mono text-sm text-primary">{callId}</p>
                    </div>
                    {wsStatus !== "connected" && wsStatus !== "connecting" ? (
                      <div className="space-y-1">
                        <p className="text-xs text-on-surface-variant font-light">Caller Phone Number</p>
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={callerNumber}
                            onChange={(e) => setCallerNumber(e.target.value)}
                            placeholder="e.g. +91 98765 43210"
                            className={`bg-background/80 border text-white rounded-lg px-3 py-1.5 font-mono text-xs focus:outline-none w-44 ${
                              findSavedContact(callerNumber)
                                ? "border-emerald-500/60 focus:border-emerald-400"
                                : "border-outline/20 focus:border-primary"
                            }`}
                          />
                          <span className="text-xs font-mono text-on-surface-variant/80 border border-outline/10 bg-surface px-2 py-1 rounded">
                            {getCountryDetails(callerNumber).flag} {getCountryDetails(callerNumber).code}
                          </span>
                        </div>
                        {(() => {
                          const match = findSavedContact(callerNumber);
                          return match ? (
                            <span className="flex items-center gap-1 text-[10px] text-emerald-400 font-semibold">
                              <CheckCircle2 className="w-3 h-3" />
                              Saved Contact: {match.name} — Analysis will be bypassed
                            </span>
                          ) : callerNumber.length > 4 ? (
                            <span className="text-[10px] text-on-surface-variant/60 font-light">
                              Unknown number — Live fraud analysis will run
                            </span>
                          ) : null;
                        })()}
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <p className="text-xs text-on-surface-variant font-light">Verified Caller Number</p>
                        <p className="font-mono text-sm text-white flex items-center gap-2">
                          {contactName ? (
                            <span className="text-emerald-400 flex items-center gap-1 font-bold">
                              <CheckCircle2 className="w-4.5 h-4.5 text-emerald-400" />
                              {contactName}
                            </span>
                          ) : (
                            <span>{callerNumber}</span>
                          )}
                          <span className="text-xs bg-surface/50 border border-outline/10 px-2 py-0.5 rounded text-primary">
                            {getCountryDetails(callerNumber).flag} {getCountryDetails(callerNumber).name}
                          </span>
                        </p>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex flex-wrap gap-3">
                    {wsStatus !== "connected" && wsStatus !== "connecting" ? (
                      <>
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={handleCaptureConnect}
                          className="flex items-center gap-2 bg-primary text-black font-headline font-bold px-6 py-3 rounded-xl text-xs shadow-lg shadow-primary/20"
                        >
                          <Mic className="w-4 h-4" /> Connect &amp; Capture Call Audio
                        </motion.button>
                        
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => connectWs(false)}
                          className="flex items-center gap-2 border border-outline/30 text-white font-headline font-bold px-6 py-3 rounded-xl text-xs hover:bg-white/5"
                        >
                          <Radio className="w-4 h-4" /> Monitor Stream Only
                        </motion.button>
                      </>
                    ) : (
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={disconnectWs}
                        className="flex items-center gap-2 bg-error/80 text-black font-headline font-bold px-6 py-3 rounded-xl text-xs"
                      >
                        <MicOff className="w-4 h-4" /> End Call
                      </motion.button>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2 shrink-0">
                    <div className={`w-2 h-2 rounded-full ${
                      wsStatus === "connected" ? "bg-primary animate-pulse" :
                      wsStatus === "connecting" ? "bg-yellow-400 animate-pulse" : "bg-outline"
                    }`} />
                    <span className="text-[10px] font-headline font-bold text-on-surface-variant uppercase tracking-widest">
                      {wsStatus}
                    </span>
                  </div>
                </div>

                {/* Instructions & Capture warnings */}
                {wsStatus !== "connected" && wsStatus !== "connecting" && (
                  <div className="bg-surface/30 rounded-xl p-4 border border-outline/10 text-xs text-on-surface-variant space-y-2">
                    <p className="font-semibold text-white flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5 text-primary" /> How to scan WhatsApp Web / Google Meet / Web Calls:
                    </p>
                    <ul className="list-disc pl-4 space-y-1 font-light">
                      <li>Click <strong>Connect & Capture Web Call</strong>.</li>
                      <li>Allow browser microphone access when prompted (captures your voice).</li>
                      <li>In the browser tab sharing dialog, select the <strong>WhatsApp Web tab</strong> (or system screen) and <strong>MUST check the &quot;Share tab audio&quot; / &quot;Share system audio&quot; option</strong> (captures caller&apos;s voice).</li>
                    </ul>
                  </div>
                )}

                {captureError && (
                  <div className="bg-error/10 border border-error/20 rounded-xl p-3.5 flex items-start gap-2.5 text-xs text-error font-light">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{captureError}</span>
                  </div>
                )}
              </div>

              {/* ── Contacts Manager ─────────────────────────────────────── */}
              {wsStatus !== "connected" && wsStatus !== "connecting" && (
                <div className="glass-panel rounded-2xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-primary" />
                      <p className="font-headline font-bold text-sm text-white">My Saved Contacts</p>
                      <span className="text-[9px] bg-primary/10 border border-primary/20 text-primary px-2 py-0.5 rounded-full font-mono">
                        {contacts.length} saved
                      </span>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.04 }}
                      onClick={() => setShowContactManager(v => !v)}
                      className="text-[10px] border border-outline/20 text-on-surface-variant px-3 py-1.5 rounded-lg hover:bg-white/5 transition-all"
                    >
                      {showContactManager ? "Hide" : "Manage Contacts"}
                    </motion.button>
                  </div>

                  {showContactManager && (
                    <div className="space-y-4 animate-fade-in">

                      {/* Status banners */}
                      {contactSuccess && (
                        <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-2.5 text-xs text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                          {contactSuccess}
                        </div>
                      )}
                      {contactError && (
                        <div className="flex items-center gap-2 bg-error/10 border border-error/20 rounded-xl px-4 py-2.5 text-xs text-error font-medium">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                          {contactError}
                        </div>
                      )}

                      {/* Add contact form */}
                      <div className="bg-surface/40 rounded-xl p-4 space-y-3 border border-outline/10">
                        <p className="text-[9px] uppercase tracking-widest text-on-surface-variant font-semibold">Add / Update Contact</p>
                        <div className="flex flex-col sm:flex-row gap-2">
                          <input
                            type="text"
                            value={newContactName}
                            onChange={e => setNewContactName(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && addContact()}
                            placeholder="Contact name (e.g. Mom)"
                            className="flex-1 bg-background/80 border border-outline/20 text-white rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-primary"
                          />
                          <input
                            type="tel"
                            value={newContactPhone}
                            onChange={e => setNewContactPhone(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && addContact()}
                            placeholder="e.g. +91 9876543210"
                            className="flex-1 bg-background/80 border border-outline/20 text-white rounded-lg px-3 py-1.5 text-xs font-mono focus:outline-none focus:border-primary"
                          />
                          <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.97 }}
                            onClick={addContact}
                            disabled={savingContact || !newContactName.trim() || !newContactPhone.trim()}
                            className="bg-primary text-black font-bold text-xs px-5 py-1.5 rounded-lg disabled:opacity-40 transition-all whitespace-nowrap"
                          >
                            {savingContact ? "Saving…" : "+ Save"}
                          </motion.button>
                        </div>
                        <p className="text-[9px] text-on-surface-variant/60 font-light">
                          Saved numbers bypass live fraud analysis. You must include the country code (e.g. +91 for India, +1 for US).
                        </p>
                      </div>

                      {/* Contact list */}
                      {contacts.length === 0 ? (
                        <div className="text-center py-6 space-y-1 text-on-surface-variant/50">
                          <User className="w-8 h-8 mx-auto opacity-20" />
                          <p className="text-xs font-light">No saved contacts yet.</p>
                          <p className="text-[10px] font-light">Add a contact above to whitelist them — they will bypass fraud analysis.</p>
                        </div>
                      ) : (
                        <div className="grid sm:grid-cols-2 gap-2">
                          {contacts.map((c) => (
                            <div key={c.phone} className="flex items-center justify-between bg-surface/30 border border-outline/10 rounded-xl px-4 py-2.5 group hover:border-outline/30 transition-all">
                              <div className="flex items-center gap-2.5">
                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary text-[11px] font-bold uppercase shrink-0">
                                  {c.name.charAt(0)}
                                </div>
                                <div className="min-w-0">
                                  <p className="text-xs text-white font-semibold truncate">{c.name}</p>
                                  <p className="text-[9px] font-mono text-on-surface-variant/70 flex items-center gap-1">
                                    <ShieldCheck className="w-2.5 h-2.5 text-emerald-500/60" />
                                    {c.phone}
                                  </p>
                                </div>
                              </div>
                              <button
                                onClick={() => removeContact(c.phone)}
                                className="opacity-0 group-hover:opacity-100 ml-2 text-[9px] text-error/70 hover:text-error border border-error/10 hover:border-error/40 px-2 py-1 rounded-md transition-all flex items-center gap-1"
                              >
                                <X className="w-2.5 h-2.5" /> Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Active Call Metadata (Duration, Country Code, Biometrics) */}
              {wsStatus === "connected" && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel rounded-2xl p-5 border border-primary/20 bg-primary/5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-primary/10 rounded-xl">
                      <Radio className="w-5 h-5 text-primary animate-pulse" />
                    </div>
                    <div>
                      <p className="text-[9px] uppercase tracking-widest text-on-surface-variant font-semibold">Active Call Session</p>
                      <h4 className="text-sm font-bold text-white flex items-center gap-2">
                        {contactName ? (
                          <span className="text-emerald-400 flex items-center gap-1">
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            {contactName} (Verified Safe)
                          </span>
                        ) : (
                          <span>Monitoring: {callerNumber}</span>
                        )}
                        <span className="text-xs px-2 py-0.5 bg-surface rounded-full text-primary border border-outline/15 font-mono font-normal">
                          {getCountryDetails(callerNumber).flag} {getCountryDetails(callerNumber).name} ({getCountryDetails(callerNumber).code})
                        </span>
                      </h4>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-6 text-xs font-mono">
                    <div className="bg-surface/50 border border-outline/10 rounded-xl px-4 py-2 flex flex-col">
                      <span className="text-[8px] uppercase tracking-wider text-on-surface-variant/80">Call Duration</span>
                      <span className="text-white font-bold text-sm tracking-widest">{formatDuration(liveDuration)}</span>
                    </div>

                    <div className="bg-surface/50 border border-outline/10 rounded-xl px-4 py-2 flex flex-col">
                      <span className="text-[8px] uppercase tracking-wider text-on-surface-variant/80">Biometric State</span>
                      <span className={contactName ? "text-emerald-400 font-semibold" : "text-cyan-400 font-semibold"}>
                        {contactName ? "Bypassed (Safe Whitelist)" : "Dual-Channel PCM"}
                      </span>
                    </div>

                    <div className="bg-surface/50 border border-outline/10 rounded-xl px-4 py-2 flex flex-col">
                      <span className="text-[8px] uppercase tracking-wider text-on-surface-variant/80">NLP Engine</span>
                      <span className={contactName ? "text-emerald-400 font-semibold" : "text-emerald-400 font-semibold animate-pulse"}>
                        {contactName ? "Secure Clearance" : "Faster-Whisper + XGB"}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Live grid: spectrogram left, score + chunks right */}
              {(wsStatus === "connected" || wsStatus === "connecting" || chunks.length > 0) && (
                <div className="space-y-6">
                  {/* Connecting / Waiting skeleton — shown before first chunk arrives */}
                  {chunks.length === 0 && (wsStatus === "connected" || wsStatus === "connecting") && (
                    <div className="glass-panel rounded-2xl border border-outline/10 p-8 flex flex-col items-center justify-center gap-4 text-center min-h-[200px]">
                      {wsStatus === "connecting" ? (
                        <>
                          <div className="flex items-center gap-3">
                            <div className="w-3 h-3 rounded-full bg-yellow-400 animate-ping" />
                            <p className="text-sm font-headline font-bold text-white">Connecting to backend…</p>
                          </div>
                          <p className="text-xs text-on-surface-variant/70 font-light max-w-sm">
                            Waiting for the analysis server at{" "}
                            <span className="text-primary font-mono text-[10px]">ws://localhost:8000</span>
                          </p>
                          <div className="text-[10px] text-yellow-400/70 bg-yellow-400/5 border border-yellow-400/20 rounded-xl px-4 py-2 max-w-sm">
                            ⚠️ If stuck here — start the backend:<br />
                            <code className="font-mono text-[9px]">cd backend → python -m uvicorn main:app --port 8000</code>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex items-center gap-3">
                            <div className="w-3 h-3 rounded-full bg-primary animate-ping" />
                            <p className="text-sm font-headline font-bold text-white">🎙️ Listening for audio…</p>
                          </div>
                          <p className="text-xs text-on-surface-variant/70 font-light max-w-sm">
                            Speak into your mic — transcript appears instantly.<br />
                            Caller audio from tab → first score in <span className="text-primary font-semibold">~1.5s</span>.
                          </p>
                          {/* Animated waveform */}
                          <div className="flex items-end gap-1 h-8">
                            {[0.4, 0.7, 1.0, 0.8, 0.5, 0.9, 0.6, 1.0, 0.7, 0.4].map((h, i) => (
                              <div
                                key={i}
                                className="w-1.5 rounded-full bg-primary/50 animate-pulse"
                                style={{ height: `${h * 32}px`, animationDelay: `${i * 0.1}s` }}
                              />
                            ))}
                          </div>
                        </>
                      )}
                      <p className="text-[9px] text-on-surface-variant/30 font-mono">
                        ws: {wsStatus} · audio: {isCapturing ? "capturing ✓" : "standby"}
                      </p>
                    </div>
                  )}

                  {/* Full results — once chunks arrive */}
                  {latestChunk && (
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Spectrogram — updates every 3rd chunk */}
                    <SpectrogramPanel
                      src={latestSpec}
                      label="Live Spectrogram"
                      live={wsStatus === "connected"}
                      chunkNumber={latestSpecChunk}
                    />

                    {/* Score panel */}
                    <div className="glass-panel rounded-2xl border border-outline/10 p-6 space-y-5">
                      <div className="flex items-center justify-between">
                        <ScoreDial score={latestChunk.current_score} />
                        <div className="space-y-2 text-right">
                          <p className={`text-2xl font-headline font-bold ${scoreColor(latestChunk.current_score)}`}>
                            {latestChunk.verdict}
                          </p>
                          <div className="flex flex-col items-end gap-0.5">
                            <span className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-headline font-bold border ${scoreBorder(latestChunk.current_score)} ${scoreColor(latestChunk.current_score)} bg-surface-high/30`}>
                              {scoreBandLabel(latestChunk.current_score, latestChunk)}
                            </span>
                            <p className="text-[10px] text-on-surface-variant/80 font-light">
                              {scoreBandSubtext(latestChunk.current_score)}
                            </p>
                          </div>
                          <div className="flex items-center gap-1.5 justify-end">
                            <Clock className="w-3 h-3 text-on-surface-variant/40" />
                            <span className="text-xs text-on-surface-variant">
                              {latestChunk.elapsed_seconds.toFixed(1)}s elapsed
                            </span>
                          </div>
                          {latestChunk.high_risk_triggered && (
                            <div className="flex items-center gap-1.5 justify-end">
                              <ShieldAlert className="w-3.5 h-3.5 text-error" />
                              <span className="text-[10px] font-bold text-error">
                                FLAGGED in {latestChunk.time_to_alert_seconds?.toFixed(2)}s
                              </span>
                            </div>
                          )}
                          {!latestChunk.high_risk_triggered && (
                            <div className="flex items-center gap-1.5 justify-end">
                              <CheckCircle2 className="w-3.5 h-3.5 text-primary/50" />
                              <span className="text-[10px] text-primary/50">No high-risk flag</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Chunk history mini-bar */}
                      <div>
                        <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-2">
                          Score history (last {chunks.length} chunks)
                        </p>
                        <div className="flex items-end gap-0.5 h-10">
                          {chunks.map((c, i) => {
                            const h = Math.max(4, (c.current_score / 100) * 40);
                            return (
                              <div
                                key={i}
                                className={`flex-1 rounded-sm transition-all ${
                                  c.current_score <= 10 ? "bg-rose-500" :
                                  c.current_score <= 25 ? "bg-red-500" :
                                  c.current_score <= 50 ? "bg-orange-400" :
                                  c.current_score <= 74 ? "bg-yellow-400" : "bg-emerald-400"
                                }`}
                                style={{ height: `${h}px` }}
                              />
                            );
                          })}
                        </div>
                      </div>

                      <div>
                        <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-2">
                          Latest reasons
                        </p>
                        <ReasonList reasons={latestChunk.reasons} />
                      </div>
                    </div>
                  </div>
                  )} {/* end latestChunk */}

                  {/* Live Transcript & Real-time Advisory — always visible when connected */}
                  <div className="grid md:grid-cols-2 gap-6 animate-fade-in">
                    {/* Live Transcript Card */}
                    <div className="glass-panel rounded-2xl border border-outline/10 p-6 flex flex-col h-[450px]">
                      <div className="flex items-center justify-between border-b border-outline/10 pb-3 mb-4">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
                          <h3 className="font-headline font-bold text-xs text-white uppercase tracking-wider">
                            Live Call Dialogue
                          </h3>
                        </div>
                        <span className="text-[9px] font-mono text-on-surface-variant font-light">
                          {liveMessages.length} statement{liveMessages.length !== 1 ? 's' : ''} captured
                        </span>
                      </div>
                      
                      {/* Chat Messages Log */}
                      <div className="flex-1 overflow-y-auto space-y-3 pr-1 scrollbar-thin">

                        {/* Live interim — always shown at top when speaking */}
                        {liveInterim && (wsStatus === "connected" || wsStatus === "connecting") && (
                          <div className="flex flex-col items-end">
                            <span className="text-[9px] font-headline font-bold uppercase tracking-wider text-emerald-400/70 mb-1">
                              🎙️ You (speaking…)
                            </span>
                            <div className="max-w-[90%] rounded-2xl px-4 py-2.5 text-xs leading-relaxed bg-emerald-500/10 border border-emerald-500/30 text-emerald-200 rounded-tr-none">
                              {liveInterim}
                              <span className="inline-block w-0.5 h-3 bg-emerald-400 ml-1 animate-pulse align-middle rounded-full" />
                            </div>
                          </div>
                        )}

                        {/* Animated listening/connecting state — only when no messages yet AND no interim */}
                        {liveMessages.length === 0 && !liveInterim && (
                          <div className="h-full flex flex-col items-center justify-center text-center gap-5 py-8">
                            {(wsStatus === "connected" || wsStatus === "connecting") ? (
                              <>
                                {/* Animated soundwave bars */}
                                <div className="flex items-end gap-1 h-10">
                                  {[0.4, 0.7, 1.0, 0.8, 0.5, 0.9, 0.6, 1.0, 0.7, 0.4].map((h, i) => (
                                    <div
                                      key={i}
                                      className="w-1.5 rounded-full bg-primary/60 animate-pulse"
                                      style={{
                                        height: `${h * 40}px`,
                                        animationDelay: `${i * 0.1}s`,
                                        animationDuration: `${0.8 + i * 0.1}s`
                                      }}
                                    />
                                  ))}
                                </div>
                                <div className="space-y-1.5">
                                  <p className="text-sm font-bold text-white flex items-center gap-2 justify-center">
                                    <span className="w-2 h-2 rounded-full bg-primary animate-ping inline-block" />
                                    Analysis Running…
                                  </p>
                                  <p className="text-[11px] text-on-surface-variant/70 font-light max-w-[240px]">
                                    Speak into your mic — your words will appear here instantly
                                  </p>
                                  <p className="text-[10px] text-on-surface-variant/40 font-mono">
                                    Caller audio from tab → backend transcript in ~1.5s
                                  </p>
                                </div>
                                {/* Quick test buttons */}
                                <div className="flex flex-col gap-1.5 w-full max-w-xs">
                                  <p className="text-[8px] uppercase tracking-widest text-on-surface-variant/50">Inject test scenario →</p>
                                  {([
                                    ["🔴 OTP Scam",        "Please share your OTP one time password to verify your account immediately"],
                                    ["🔴 Digital Arrest",  "This is CBI officer you are under digital arrest for money laundering court order issued"],
                                    ["🔴 AnyDesk",         "Download AnyDesk for remote access so I can fix your computer virus right now"],
                                    ["🔴 Lottery",         "Congratulations you have won lottery prize money pay processing fee to claim prize"],
                                    ["🟢 Safe Call",       "Hello I am calling to confirm your appointment for tomorrow at 3pm"],
                                    ["🟢 Banking Alert",   "For your safety, the bank will never ask you to share your OTP or money. Keep your details safe."],
                                    ["🟢 Standard Pay",    "I am returning the cash that I borrowed yesterday for my grocery shopping."],
                                  ] as [string, string][]).map(([label, txt]) => (
                                    <button
                                      key={label}
                                      onClick={async () => {
                                        setManualSpeaker("Caller");
                                        setManualText(txt);
                                        setTimeout(() => {
                                          const form = document.getElementById("inject-form") as HTMLFormElement | null;
                                          form?.requestSubmit();
                                        }, 60);
                                      }}
                                      className="text-[10px] text-left px-3 py-1.5 rounded-lg border border-outline/15 hover:border-primary/30 hover:bg-primary/5 text-on-surface-variant/70 hover:text-white transition-all"
                                    >
                                      {label}
                                    </button>
                                  ))}
                                </div>
                              </>
                            ) : (
                              <div className="flex flex-col items-center gap-2 opacity-40">
                                <Radio className="w-8 h-8 text-on-surface-variant/40" />
                                <p className="text-[11px] text-on-surface-variant font-light">
                                  Connect a call to begin transcript
                                </p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Finalized messages */}
                        {liveMessages.map((msg, idx) => {
                          const isYou = msg.speaker === "You";
                          return (
                            <div key={idx} className={`flex flex-col ${isYou ? "items-end" : "items-start"}`}>
                              <div className="flex items-center gap-1.5 mb-1">
                                <span className={`text-[9px] font-headline font-bold uppercase tracking-wider ${
                                  isYou ? "text-emerald-400" : "text-cyan-400"
                                }`}>
                                  {isYou ? "🙋 You (Client)" : "📞 Caller (Person 1)"}
                                </span>
                                <span className="text-[8px] text-on-surface-variant/50 font-light">{msg.timestamp}</span>
                              </div>
                              <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-xs leading-relaxed ${
                                isYou
                                  ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-100 rounded-tr-none"
                                  : "bg-surface-high border border-outline/10 text-white rounded-tl-none"
                              }`}>
                                {highlightKeywords(msg.text)}
                              </div>
                            </div>
                          );
                        })}

                        <div ref={chatEndRef} />
                      </div>

                      {/* Manual Simulation Transcript Injector */}
                      {wsStatus === "connected" && (
                        <form id="inject-form" onSubmit={injectManualTranscript} className="border-t border-outline/10 pt-3 mt-3 flex flex-col gap-2">
                          <p className="text-[8px] uppercase tracking-wider text-on-surface-variant/80 font-bold mb-1">
                            🚨 Simultaneously Inject Simulated Call Statement
                          </p>
                          <div className="flex gap-2">
                            <select
                              value={manualSpeaker}
                              onChange={(e) => setManualSpeaker(e.target.value as "You" | "Caller")}
                              className="bg-surface border border-outline/20 text-white rounded-xl px-2 py-1 text-xs focus:outline-none cursor-pointer"
                            >
                              <option value="Caller">Caller</option>
                              <option value="You">Client (You)</option>
                            </select>
                            <input
                              type="text"
                              value={manualText}
                              onChange={(e) => setManualText(e.target.value)}
                              placeholder={`Type simulated ${manualSpeaker === "Caller" ? "fraud/scam statement" : "response"}...`}
                              className="flex-1 bg-surface-high border border-outline/20 text-white rounded-xl px-3 py-1 text-xs focus:outline-none focus:border-primary/50"
                            />
                            <motion.button
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                              type="submit"
                              disabled={injectingText || !manualText.trim()}
                              className="bg-primary/20 hover:bg-primary border border-primary/30 hover:border-primary text-primary hover:text-black font-semibold text-xs px-4 py-1.5 rounded-xl transition-all disabled:opacity-40"
                            >
                              {injectingText ? "Injecting..." : "Inject"}
                            </motion.button>
                          </div>
                        </form>
                      )}
                    </div>

                    {/* Live Advisory Card */}
                    {(() => {
                      const score = latestChunk.current_score;
                      const intent = latestChunk.nlp_intent ?? "unknown";
                      const advisory = contactName
                        ? {
                            status: "SAFE",
                            color: "text-emerald-400",
                            bgColor: "bg-emerald-500/5",
                            borderColor: "border-emerald-500/30",
                            icon: <Shield className="w-5 h-5 text-emerald-400 shrink-0" />,
                            heading: "WHITELIST VERIFIED: SECURE CONTACT",
                            description: `This caller is registered on your Whitelist as '${contactName}'. Threat assessment has been bypassed because their phone identity is trusted and secure.`,
                            actions: [
                              `🟢 Identity Verified: '${contactName}' (${callerNumber})`,
                              "🛡️ General security monitoring bypassed for this safe caller.",
                              "⚠️ Standard reminder: If this contact behaves unexpectedly or requests credentials, verify them via an alternate channel."
                            ]
                          }
                        : getAdvisory(
                            score,
                            intent,
                            <ShieldAlert className="w-5 h-5 text-red-500 shrink-0 animate-bounce" />,
                            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 animate-pulse" />,
                            <Shield className="w-5 h-5 text-emerald-500 shrink-0" />
                          );
                      
                      return (
                        <div className={`glass-panel rounded-2xl border border-outline/10 p-6 flex flex-col justify-between h-[450px] ${advisory.bgColor} border-2 ${advisory.borderColor} transition-all duration-500`}>
                          <div className="space-y-4">
                            <div className="flex items-center gap-2.5 border-b border-outline/10 pb-3">
                              {advisory.icon}
                              <h3 className="font-headline font-bold text-xs text-white uppercase tracking-wider">
                                Safety Advisory System
                              </h3>
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-baseline gap-2">
                                <span className={`text-xs font-bold uppercase ${advisory.color}`}>
                                  {advisory.heading}
                                </span>
                              </div>
                              <p className="text-[11px] text-on-surface-variant leading-relaxed font-light">
                                {advisory.description}
                              </p>
                            </div>

                            <div className="space-y-2 flex-1">
                              <p className="text-[9px] uppercase tracking-widest text-on-surface-variant font-semibold mb-2">
                                Recommended Protective Actions:
                              </p>
                              <div className="space-y-2 overflow-y-auto max-h-[140px] pr-1 scrollbar-thin">
                                {advisory.actions.map((act, i) => (
                                  <div
                                    key={i}
                                    className="flex items-start gap-2 text-[10px] text-white leading-normal bg-black/20 rounded-lg p-2 border border-outline/5"
                                  >
                                    <span className="shrink-0">•</span>
                                    <span>{act}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>

                          <div className="text-[8px] text-on-surface-variant/40 text-center font-mono mt-3">
                            Operation Safe Vault • Real-time NLP Analysis Active
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}

              {(wsStatus === "idle" || wsStatus === "ended") && (
                <div className="flex flex-col items-center gap-3 py-16 border-2 border-dashed border-outline/10 rounded-3xl">
                  <AlertTriangle className="w-10 h-10 text-on-surface-variant/20" />
                  <p className="text-on-surface-variant/40 text-sm font-light">
                    {wsStatus === "ended" ? "Live call session ended. Click Connect to start a new session." : "Click Connect to open a live analysis WebSocket session"}
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
