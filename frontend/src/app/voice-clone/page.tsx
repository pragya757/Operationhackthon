"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  ShieldAlert,
  ShieldCheck,
  Loader2,
  AlertTriangle,
  FileAudio,
  Brain,
  Cpu,
  Activity,
  CheckCircle2,
  Mic,
  Square,
  Radio,
  PhoneOff,
  Zap,
  Clock,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Navbar } from "@/components/Navbar";

// ── Types ─────────────────────────────────────────────────────────────────────
interface ForensicAnalysis {
  prediction: string;
  prediction_raw?: string;
  confidence: number;
  risk_level?: string;
  reasons: string[];
  spectrogram_image?: string | null;
  forensic_note?: string | null;
}

interface ThreatFusionResult {
  final_risk_score: number;
  risk_level: string;
  explanation: string[];
}

interface ForensicsResult {
  prediction: string;
  prediction_internal: string;
  confidence: number;
  model_score: number;
  heuristic_score: number;
  fusion_score: number;
  risk_level: string;
  reasons: string[];
  spectrogram_image: string | null;
  voice_clone_analysis?: ForensicAnalysis;
  spectrogram_analysis?: ForensicAnalysis;
  threat_fusion?: ThreatFusionResult;
}

interface LiveForensicsChunk extends ForensicsResult {
  chunk_index: number;
  elapsed_seconds: number;
}

// ── Constants ─────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_BASE  = API_BASE.replace(/^http/, "ws");

// ── WAV encoder (identical to Voice Lab) ─────────────────────────────────────
function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function encodeWAV(buffers: Float32Array[], sampleRate: number, numChannels: number = 1): ArrayBuffer {
  const bytesPerSample = 2;
  const numSamples     = buffers.reduce((acc, b) => acc + b.length, 0);
  const buffer         = new ArrayBuffer(44 + numSamples * bytesPerSample);
  const view           = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + numSamples * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1,          true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate,  true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, numSamples * bytesPerSample, true);

  let offset = 44;
  for (const b of buffers) {
    for (let i = 0; i < b.length; i++) {
      const s = Math.max(-1, Math.min(1, b[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
  }
  return buffer;
}

// ── Helper colours ────────────────────────────────────────────────────────────
function getRiskBadgeColor(risk: string) {
  if (risk === "High Risk") return "bg-red-500/10 text-red-500 border border-red-500/30 shadow-[0_0_12px_rgba(239,68,68,0.2)]";
  if (risk === "Suspicious") return "bg-amber-500/10 text-amber-500 border border-amber-500/30 shadow-[0_0_12px_rgba(245,158,11,0.2)]";
  return "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.2)]";
}
function getRiskIcon(risk: string) {
  if (risk === "High Risk") return <ShieldAlert className="w-5 h-5 text-red-500" />;
  if (risk === "Suspicious") return <AlertTriangle className="w-5 h-5 text-amber-500" />;
  return <ShieldCheck className="w-5 h-5 text-emerald-500" />;
}
function getPredictionColor(pred: string) {
  if (pred === "Human") return "text-emerald-500";
  if (pred === "AI Generated") return "text-red-500 font-bold";
  if (pred === "Suspicious Acoustic Pattern") return "text-amber-400 font-bold";
  return "text-red-500 font-bold";
}
function getRiskRingColor(risk: string) {
  if (risk === "High Risk") return "stroke-red-500";
  if (risk === "Suspicious") return "stroke-amber-500";
  return "stroke-emerald-500";
}
function fmtDuration(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

// ── Reusable Threat Gauge ─────────────────────────────────────────────────────
function ThreatGauge({ score, risk, size = 144 }: { score: number; risk: string; size?: number }) {
  const r  = size / 2 - 10;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg className="w-full h-full -rotate-90" viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} className="stroke-surface/40 fill-none" strokeWidth="8" />
        <motion.circle
          cx={cx} cy={cy} r={r}
          className={`fill-none ${getRiskRingColor(risk)}`}
          strokeWidth="8"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ * (1 - score / 100) }}
          transition={{ duration: 1.0, ease: "easeOut" }}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-2xl font-extrabold font-headline tracking-tighter text-white">
          {score}%
        </span>
        <span className="text-[9px] text-on-surface-variant uppercase tracking-widest font-bold font-headline">
          Threat Score
        </span>
      </div>
    </div>
  );
}

// ── Forensics Result Panel (shared by Upload + Live final report) ─────────────
function ForensicsResultPanel({ result }: { result: ForensicsResult }) {
  if (!result) return null;

  // Handle case where forensics were skipped (e.g. ended early)
  if ((result as any).forensics_skipped) {
    return (
      <div className="bg-surface/20 backdrop-blur-xl border border-outline/10 rounded-2xl p-6 sm:p-8 space-y-6">
        <div className="flex items-center gap-3 text-amber-500 font-headline font-bold">
          <AlertTriangle className="w-6 h-6" />
          <span>Voice Forensics Skipped</span>
        </div>
        <p className="text-sm text-on-surface-variant font-headline font-light bg-surface/30 p-4 rounded-xl border border-outline/5 leading-relaxed">
          {(result as any).message || "The recording session was ended before the required 10-second threshold. A minimum of 10 seconds of continuous audio is required to perform accurate AI Voice Clone and Spectrogram analysis."}
        </p>
      </div>
    );
  }

  // Handle preview update object (does not have prediction or threat_fusion yet)
  if (!result.prediction && !result.threat_fusion) {
    return null;
  }

  return (
    <div className="bg-surface/20 backdrop-blur-xl border border-outline/10 rounded-2xl p-6 sm:p-8 space-y-8">
      {result.threat_fusion ? (
        <>
          {/* Final Threat Assessment */}
          <div className="border-b border-outline/15 pb-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
              <div>
                <p className="text-[10px] text-primary font-bold uppercase tracking-widest mb-1 font-headline">Final Threat Assessment</p>
                <h2 className="text-3xl font-extrabold font-headline tracking-tight text-white flex items-center gap-2">
                  Combined Risk:{" "}
                  <span className={getPredictionColor(result.threat_fusion.risk_level === "Safe" ? "Human" : result.threat_fusion.risk_level === "Suspicious" ? "AI Generated" : "Voice Clone")}>
                    {result.threat_fusion.risk_level}
                  </span>
                </h2>
              </div>
              <div className={`px-4 py-1.5 rounded-full text-xs font-bold font-headline uppercase tracking-wider flex items-center gap-1.5 self-start sm:self-center ${getRiskBadgeColor(result.threat_fusion.risk_level)}`}>
                {getRiskIcon(result.threat_fusion.risk_level)}
                {result.threat_fusion.risk_level}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
              <div className="md:col-span-4 flex flex-col items-center justify-center">
                <ThreatGauge score={result.threat_fusion.final_risk_score} risk={result.threat_fusion.risk_level} />
              </div>
              <div className="md:col-span-8 space-y-3 bg-surface/30 p-5 rounded-xl border border-outline/10 font-headline">
                <h4 className="text-white text-xs font-bold uppercase tracking-wider">Combined Forensic Explanation</h4>
                <div className="space-y-2">
                  {result.threat_fusion.explanation.map((reason, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 text-sm font-light text-on-surface-variant bg-surface/20 px-3 py-2 rounded-lg border border-outline/5">
                      <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0 text-primary" />
                      <span>{reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Engine Breakdowns */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 border-b border-outline/15 pb-6">
            {/* Voice Clone */}
            <div className="space-y-4">
              <h3 className="text-white font-headline font-semibold text-lg flex items-center gap-2 border-b border-outline/5 pb-2">
                <Brain className="w-5 h-5 text-primary" /> Voice Clone Detection
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-surface/30 p-3 rounded-lg border border-outline/5">
                  <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">Prediction</span>
                  <span className={`text-sm font-bold ${getPredictionColor(result.voice_clone_analysis?.prediction || "")}`}>
                    {result.voice_clone_analysis?.prediction}
                  </span>
                </div>
                <div className="bg-surface/30 p-3 rounded-lg border border-outline/5">
                  <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">Confidence</span>
                  <span className="text-sm font-semibold text-white font-mono">{result.voice_clone_analysis?.confidence}%</span>
                </div>
              </div>
              <div className="space-y-2">
                <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">Acoustic Insights</span>
                {result.voice_clone_analysis?.reasons.map((r, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs font-light text-on-surface-variant">
                    <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-primary" />
                    <span>{r}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Spectrogram Forensic Evidence */}
            <div className="space-y-4">
              <h3 className="text-white font-headline font-semibold text-lg flex items-center gap-2 border-b border-outline/5 pb-2">
                <Cpu className="w-5 h-5 text-primary" /> Spectrogram Forensic Evidence
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-surface/30 p-3 rounded-lg border border-outline/5">
                  <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">CNN Finding</span>
                  <span className={`text-sm font-bold ${getPredictionColor(result.spectrogram_analysis?.prediction || "")}`}>
                    {result.spectrogram_analysis?.prediction}
                  </span>
                  {result.spectrogram_analysis?.prediction === "Suspicious Acoustic Pattern" && (
                    <span className="mt-1 inline-flex items-center gap-1 text-[9px] font-bold font-headline uppercase tracking-wider text-amber-400/80 bg-amber-400/10 px-1.5 py-0.5 rounded-full">
                      🔬 Forensic Indicator
                    </span>
                  )}
                </div>
                <div className="bg-surface/30 p-3 rounded-lg border border-outline/5">
                  <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">CNN Score</span>
                  <span className="text-sm font-semibold text-white font-mono">{result.spectrogram_analysis?.confidence}%</span>
                </div>
              </div>
              {/* Domain gap note */}
              {result.spectrogram_analysis?.forensic_note && (
                <p className="text-[10px] italic text-on-surface-variant/50 bg-surface/20 px-3 py-2 rounded-lg border border-outline/5 leading-relaxed">
                  ℹ️ {result.spectrogram_analysis.forensic_note}
                </p>
              )}
              <div className="space-y-2">
                <span className="text-[10px] text-on-surface-variant block uppercase font-bold tracking-wider font-headline">Detected Artifacts & Explainability</span>
                {result.spectrogram_analysis?.reasons.map((r, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs font-light text-on-surface-variant">
                    <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-primary" />
                    <span>{r}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Spectrogram image */}
          {result.spectrogram_analysis?.spectrogram_image && (
            <div className="space-y-3">
              <h4 className="text-white font-headline text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                Mel-Spectrogram Signature
              </h4>
              <div className="bg-[#131313] p-3 rounded-xl border border-outline/10 overflow-hidden flex items-center justify-center">
                <img
                  src={result.spectrogram_analysis.spectrogram_image}
                  alt="Mel Spectrogram"
                  className="max-h-64 object-contain w-full rounded-lg hover:scale-[1.01] transition-transform duration-300"
                />
              </div>
            </div>
          )}
        </>
      ) : (
        /* Legacy single-engine result */
        <>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-outline/15 pb-6">
            <div>
              <p className="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest mb-1">Forensic Verdict</p>
              <h2 className="text-3xl font-extrabold font-headline tracking-tight text-white flex items-center gap-2">
                Classified as: <span className={getPredictionColor(result.prediction)}>{result.prediction}</span>
              </h2>
            </div>
            <div className={`px-4 py-1.5 rounded-full text-xs font-bold font-headline uppercase tracking-wider flex items-center gap-1.5 self-start sm:self-center ${getRiskBadgeColor(result.risk_level)}`}>
              {getRiskIcon(result.risk_level)} {result.risk_level}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="flex flex-col items-center justify-center">
              <div className="relative w-40 h-40">
                <svg className="w-full h-full transform -rotate-90">
                  <circle cx="80" cy="80" r="68" className="stroke-surface/40 fill-none" strokeWidth="8" />
                  <motion.circle cx="80" cy="80" r="68" className={`fill-none ${getRiskRingColor(result.risk_level)}`} strokeWidth="8" strokeDasharray={2 * Math.PI * 68}
                    initial={{ strokeDashoffset: 2 * Math.PI * 68 }}
                    animate={{ strokeDashoffset: 2 * Math.PI * 68 * (1 - result.confidence / 100) }}
                    transition={{ duration: 1.2, ease: "easeOut" }} strokeLinecap="round" />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-3xl font-extrabold font-headline tracking-tighter text-white">{result.confidence}%</span>
                  <span className="text-[10px] text-on-surface-variant uppercase tracking-widest font-bold font-headline">Confidence</span>
                </div>
              </div>
            </div>
            <div className="space-y-4 font-headline bg-surface/30 p-4 rounded-xl border border-outline/10">
              <h4 className="text-white text-xs font-bold uppercase tracking-wider mb-2">Metrics Overview</h4>
              <div className="flex justify-between items-center text-xs py-1.5 border-b border-outline/5">
                <span className="text-on-surface-variant flex items-center gap-1.5"><Brain className="w-3.5 h-3.5 text-primary" /> Model Score</span>
                <span className="font-mono text-white font-semibold">{(result.model_score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center text-xs py-1.5 border-b border-outline/5">
                <span className="text-on-surface-variant flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-primary" /> Heuristic Score</span>
                <span className="font-mono text-white font-semibold">{(result.heuristic_score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center text-xs py-1.5">
                <span className="text-on-surface-variant flex items-center gap-1.5 font-bold"><Activity className="w-3.5 h-3.5 text-primary animate-pulse" /> Final Fusion Score</span>
                <span className="font-mono text-primary font-bold">{(result.fusion_score * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="space-y-3">
            <h4 className="text-white font-headline text-xs font-bold uppercase tracking-wider">Acoustic & Deep Learning Insights</h4>
            <div className="space-y-2">
              {result.reasons.map((r, i) => (
                <div key={i} className="flex items-start gap-2.5 text-sm font-light text-on-surface-variant bg-surface/20 px-3.5 py-2.5 rounded-lg border border-outline/5 hover:border-primary/10 transition-colors">
                  <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0 text-primary" /> <span>{r}</span>
                </div>
              ))}
            </div>
          </div>
          {result.spectrogram_image && (
            <div className="space-y-3">
              <h4 className="text-white font-headline text-xs font-bold uppercase tracking-wider">Mel-Spectrogram Signature</h4>
              <div className="bg-[#131313] p-3 rounded-xl border border-outline/10 overflow-hidden flex items-center justify-center">
                <img src={result.spectrogram_image} alt="Mel Spectrogram" className="max-h-60 object-contain w-full rounded-lg hover:scale-[1.02] transition-transform duration-300" />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Live Dashboard Panel ──────────────────────────────────────────────────────
function LiveDashboard({
  status,
  callId,
  duration,
  latest,
  history,
  captureError,
  onDisconnect,
}: {
  status: "connecting" | "connected" | "ended";
  callId: string;
  duration: number;
  latest: LiveForensicsChunk | null;
  history: LiveForensicsChunk[];
  captureError: string | null;
  onDisconnect: () => void;
}) {
  const risk = latest?.threat_fusion?.risk_level ?? "Safe";
  const score = latest?.threat_fusion?.final_risk_score ?? 0;

  return (
    <div className="bg-surface/20 backdrop-blur-xl border border-outline/10 rounded-2xl overflow-hidden">
      {/* Status bar */}
      <div className={`px-6 py-3 flex items-center justify-between border-b border-outline/10 ${
        status === "connected" ? "bg-primary/5" : status === "ended" ? "bg-surface/30" : "bg-surface/30"
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${
            status === "connected" ? "bg-primary animate-pulse" :
            status === "connecting" ? "bg-amber-400 animate-pulse" : "bg-outline"
          }`} />
          <span className="text-xs font-bold uppercase tracking-widest font-headline text-on-surface-variant">
            {status === "connected" ? "Live · Voice Forensics Active" :
             status === "connecting" ? "Connecting..." : "Session Ended"}
          </span>
        </div>
        <div className="flex items-center gap-4">
          {status === "connected" && (
            <div className="flex items-center gap-1.5 text-xs font-mono text-on-surface-variant">
              <Clock className="w-3.5 h-3.5" />
              {fmtDuration(duration)}
            </div>
          )}
          {(status === "connected" || status === "connecting") && (
            <button
              onClick={onDisconnect}
              className="flex items-center gap-1.5 text-xs font-bold text-red-400 hover:text-red-300 border border-red-500/20 hover:border-red-500/50 px-3 py-1 rounded-lg transition-all font-headline"
            >
              <PhoneOff className="w-3.5 h-3.5" /> End Call
            </button>
          )}
        </div>
      </div>

      {/* Session info row */}
      <div className="px-6 py-3 flex items-center gap-6 border-b border-outline/10 bg-surface/10 text-[10px] text-on-surface-variant font-mono">
        <span>SESSION <span className="text-primary">{callId}</span></span>
        {latest && (
          <span>CHUNK <span className="text-white">{latest.chunk_index}</span> · {latest.elapsed_seconds}s</span>
        )}
        <span className="ml-auto">
          {status === "connected" ? (
            <span className="flex items-center gap-1 text-primary"><Wifi className="w-3 h-3" /> STREAM ACTIVE</span>
          ) : status === "ended" ? (
            <span className="flex items-center gap-1"><WifiOff className="w-3 h-3" /> STREAM CLOSED</span>
          ) : (
            <span className="flex items-center gap-1 text-amber-400"><Loader2 className="w-3 h-3 animate-spin" /> WAITING</span>
          )}
        </span>
      </div>

      <div className="p-6 space-y-6">
        {captureError && (
          <div className="bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs p-3 rounded-xl flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{captureError}</span>
          </div>
        )}

        {/* Connecting skeleton */}
        {status === "connecting" && !latest && (
          <div className="flex flex-col items-center justify-center py-12 gap-4">
            <Loader2 className="w-10 h-10 text-primary animate-spin" />
            <p className="text-white font-headline font-semibold">Establishing audio stream…</p>
            <p className="text-on-surface-variant text-xs">Voice Forensics will begin analysing after the first 4 seconds</p>
          </div>
        )}

        {/* Waiting for first chunk */}
        {status === "connected" && !latest && (
          <div className="flex flex-col items-center justify-center py-12 gap-4">
            <div className="relative">
              <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
              <div className="p-4 rounded-full bg-primary/10 border border-primary/20 text-primary relative z-10">
                <Mic className="w-8 h-8 animate-pulse" />
              </div>
            </div>
            <p className="text-white font-headline font-semibold">Capturing audio…</p>
            <p className="text-on-surface-variant text-xs">First forensic result arrives after 4 seconds of call audio</p>
          </div>
        )}

        {/* Active Recording UI */}
        {status === "connected" && (
          <div className="flex flex-col items-center justify-center py-6 gap-6 w-full max-w-lg mx-auto">
            {/* Progress Bar and Pulsing text */}
            <div className="w-full space-y-3">
              <div className="flex justify-between items-center text-xs font-headline font-bold text-on-surface-variant uppercase tracking-wider">
                {duration < 10 ? (
                  <>
                    <span className="flex items-center gap-1.5 animate-pulse text-primary">
                      <Mic className="w-3.5 h-3.5 text-primary" /> Recording... {fmtDuration(duration)}
                    </span>
                    <span className="text-on-surface-variant/70 font-medium">Analyzing live waveform...</span>
                  </>
                ) : (
                  <>
                    <span className="flex items-center gap-1.5 text-primary font-bold">
                      <CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Recording Complete
                    </span>
                    <span className="flex items-center gap-1.5 text-primary animate-pulse font-bold">
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" /> Analyzing...
                    </span>
                  </>
                )}
              </div>
              <div className="h-2 w-full bg-surface/40 rounded-full border border-outline/10 overflow-hidden">
                <motion.div
                  className="h-full bg-primary"
                  initial={{ width: "0%" }}
                  animate={{ width: `${Math.min(100, (duration / 10) * 100)}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>
              <p className="text-[10px] text-center text-on-surface-variant/50 font-light">
                {duration < 10
                  ? "Collecting call audio for deepfake analysis. Continuous real-time waveform visualization active."
                  : "Processing complete 10-second capture. Running Voice Clone and Spectrogram CNN detection..."}
              </p>
            </div>

            {/* Optional spectrogram preview */}
            {latest?.spectrogram_analysis?.spectrogram_image ? (
              <div className="w-full space-y-2">
                <div className="flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-primary animate-pulse" />
                  <span className="text-[10px] text-primary font-bold uppercase tracking-widest font-headline">Live Mel-Spectrogram</span>
                </div>
                <div className="bg-[#131313] p-3 rounded-xl border border-outline/10 overflow-hidden flex items-center justify-center">
                  <motion.img
                    key={latest.chunk_index}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    src={latest.spectrogram_analysis.spectrogram_image}
                    alt="Live Mel Spectrogram"
                    className="w-full rounded-lg object-contain max-h-48"
                  />
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-6 gap-3 border border-dashed border-outline/10 rounded-xl w-full">
                <Loader2 className="w-6 h-6 text-on-surface-variant/20 animate-spin" />
                <span className="text-[10px] text-on-surface-variant/30 font-light">Awaiting first spectrogram frame...</span>
              </div>
            )}
          </div>
        )}

        {/* Ended: show final report link */}
        {status === "ended" && latest && (
          <div className="border-t border-outline/10 pt-4">
            <p className="text-[10px] text-primary font-bold uppercase tracking-widest font-headline mb-3">Final Forensic Report</p>
            <ForensicsResultPanel result={latest} />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function VoiceClonePage() {
  // ── Tab ──────────────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"upload" | "live">("upload");

  // ── Upload State ─────────────────────────────────────────────────────────────
  const [file, setFile]               = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState<ForensicsResult | null>(null);
  const [error, setError]             = useState<string | null>(null);
  const fileInputRef                  = useRef<HTMLInputElement>(null);

  // ── Live Stream State ────────────────────────────────────────────────────────
  const [lsStatus, setLsStatus]         = useState<"idle" | "connecting" | "connected" | "ended">("idle");
  const [lsCallId, setLsCallId]         = useState("");
  const [lsDuration, setLsDuration]     = useState(0);
  const [lsLatest, setLsLatest]         = useState<LiveForensicsChunk | null>(null);
  const [lsHistory, setLsHistory]       = useState<LiveForensicsChunk[]>([]);
  const [lsCaptureError, setLsCaptureError] = useState<string | null>(null);

  // ── Live Stream Refs ─────────────────────────────────────────────────────────
  const lsWsRef         = useRef<WebSocket | null>(null);
  const lsAudioCtxRef   = useRef<AudioContext | null>(null);
  const lsProcessorRef  = useRef<ScriptProcessorNode | null>(null);
  const lsStreamsRef    = useRef<MediaStream[]>([]);
  const lsSampleBufRef  = useRef<Float32Array[]>([]);
  const lsSendIntervalRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const lsDurationTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Cleanup on unmount ───────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      cleanupLiveStream();
      lsWsRef.current?.close();
    };
  }, []);

  // ── Upload helpers ───────────────────────────────────────────────────────────
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragActive(e.type === "dragenter" || e.type === "dragover");
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files?.[0]) validateAndSetFile(e.dataTransfer.files[0]);
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) validateAndSetFile(e.target.files[0]);
  };
  const validateAndSetFile = (f: File) => {
    setError(null);
    const ext = f.name.substring(f.name.lastIndexOf(".")).toLowerCase();
    if (![".wav", ".mp3", ".m4a", ".flac"].includes(ext)) {
      setError("Unsupported format. Supported: .wav, .mp3, .m4a, .flac"); return;
    }
    if (f.size > 50 * 1024 * 1024) { setError("File exceeds 50 MB limit."); return; }
    setFile(f);
  };
  const clearFile = () => { setFile(null); setResult(null); setError(null); };
  const runAnalysis = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData(); fd.append("file", file);
    try {
      const resp = await fetch(`${API_BASE}/api/voice-clone/analyze`, { method: "POST", body: fd });
      if (!resp.ok) { const d = await resp.json(); throw new Error(d.detail || "Forensic analysis failed."); }
      setResult(await resp.json());
    } catch (err: any) {
      setError(err.message || "Unexpected error during analysis.");
    } finally {
      setLoading(false);
    }
  };

  // ── Live Stream: cleanup audio ───────────────────────────────────────────────
  const cleanupLiveStream = () => {
    if (lsSendIntervalRef.current)  { clearInterval(lsSendIntervalRef.current);  lsSendIntervalRef.current  = null; }
    if (lsDurationTimerRef.current) { clearInterval(lsDurationTimerRef.current); lsDurationTimerRef.current = null; }
    if (lsProcessorRef.current)     { lsProcessorRef.current.disconnect(); lsProcessorRef.current = null; }
    lsStreamsRef.current.forEach(s => s.getTracks().forEach(t => t.stop()));
    lsStreamsRef.current = [];
    if (lsAudioCtxRef.current) { lsAudioCtxRef.current.close().catch(() => {}); lsAudioCtxRef.current = null; }
    lsSampleBufRef.current = [];
  };

  // ── Live Stream: connect ─────────────────────────────────────────────────────
  const connectLiveStream = async () => {
    setLsCaptureError(null);
    setLsLatest(null);
    setLsHistory([]);
    setLsDuration(0);

    const id = `fsc-${Date.now().toString(36)}`;
    setLsCallId(id);
    setLsStatus("connecting");
    console.log(`[LiveStream] ${id}: Starting connection sequence`);

    // Step 1: Mic
    let micStream: MediaStream | null = null;
    let tabStream: MediaStream | null = null;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      console.log(`[LiveStream] ${id}: ✅ Mic GRANTED —`, micStream.getAudioTracks().map(t => t.label));
    } catch (micErr) {
      console.warn(`[LiveStream] ${id}: ⚠️ Mic denied —`, micErr);
    }

    // Step 2: Tab capture (caller audio)
    try {
      tabStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "browser" } as any,
        audio: true,
      });
      const audioTracks = tabStream.getAudioTracks();
      if (audioTracks.length === 0) {
        console.warn(`[LiveStream] ${id}: ⚠️ Tab shared but NO audio tracks — tick "Share tab audio"`);
        setLsCaptureError("Tab shared but without audio — tick \"Share tab audio\" in the browser dialog. Mic-only mode active.");
        tabStream.getTracks().forEach(t => t.stop());
        tabStream = null;
      } else {
        console.log(`[LiveStream] ${id}: ✅ Tab audio GRANTED —`, audioTracks.map(t => t.label));
        audioTracks[0].addEventListener("ended", () => {
          console.warn(`[LiveStream] ${id}: ⚠️ Tab audio track ENDED (user stopped screen share)`);
          setLsCaptureError("Screen share stopped — mic-only mode continuing.");
        });
      }
    } catch (tabErr) {
      if (micStream) {
        console.warn(`[LiveStream] ${id}: ⚠️ Tab declined — mic-only mode`, tabErr);
        setLsCaptureError("Tab sharing declined — capturing mic only. For full analysis, share the call tab with audio.");
      } else {
        console.warn(`[LiveStream] ${id}: ⚠️ No audio at all`, tabErr);
        setLsCaptureError("No audio permissions granted. Stream open but no audio will be sent.");
      }
    }

    // Step 3: Open WebSocket — same endpoint as Voice Lab
    const wsUrl = `${WS_BASE}/ws/production-live-call/${id}`;
    console.log(`[LiveStream] ${id}: Opening WebSocket → ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    lsWsRef.current = ws;

    // Local flag avoids stale React state reads in ws.onclose closure
    let sessionEnded = false;

    ws.onopen = async () => {
      console.log(`[LiveStream] ${id}: ✅ WebSocket CONNECTED (readyState=${ws.readyState})`);
      setLsStatus("connected");

      // Duration timer
      let secs = 0;
      lsDurationTimerRef.current = setInterval(() => {
        secs += 1;
        setLsDuration(secs);
        if (secs >= 10) {
          console.log(`[LiveStream] ${id}: Auto-stop reached 10 seconds. Disconnecting stream.`);
          disconnectLiveStream();
        }
      }, 1000);
      console.log(`[LiveStream] ${id}: Duration timer started`);

      // Audio capture
      if (micStream || tabStream) {
        const streams: MediaStream[] = [];
        if (micStream) streams.push(micStream);
        if (tabStream) streams.push(tabStream);
        lsStreamsRef.current = streams;

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioCtx({ sampleRate: 16000 });
        lsAudioCtxRef.current = ctx;
        console.log(`[LiveStream] ${id}: AudioContext — sampleRate=${ctx.sampleRate} state=${ctx.state}`);

        const merger = ctx.createChannelMerger(2);
        const processor = ctx.createScriptProcessor(4096, 2, 2);
        lsProcessorRef.current = processor;

        if (micStream) {
          ctx.createMediaStreamSource(micStream).connect(merger, 0, 0);
          console.log(`[LiveStream] ${id}: Mic → merger ch0`);
        }
        if (tabStream) {
          ctx.createMediaStreamSource(new MediaStream(tabStream.getAudioTracks())).connect(merger, 0, 1);
          console.log(`[LiveStream] ${id}: Tab → merger ch1`);
        }

        merger.connect(processor);
        const silence = ctx.createGain();
        silence.gain.value = 0;
        processor.connect(silence);
        silence.connect(ctx.destination);
        console.log(`[LiveStream] ${id}: Audio graph: [mic/tab] → merger → processor → silence`);

        processor.onaudioprocess = (ev) => {
          const left  = ev.inputBuffer.getChannelData(0);
          const right = ev.inputBuffer.getChannelData(1);
          const interleaved = new Float32Array(left.length + right.length);
          for (let i = 0; i < left.length; i++) {
            interleaved[i * 2]     = left[i];
            interleaved[i * 2 + 1] = right[i];
          }
          lsSampleBufRef.current.push(interleaved);
        };

        // Send 1-second stereo WAV chunks to backend (identical to Voice Lab)
        let chunksSent = 0;
        lsSendIntervalRef.current = setInterval(() => {
          const bufs = lsSampleBufRef.current;
          if (!bufs.length) {
            console.log(`[LiveStream] ${id}: 1s tick — no samples buffered yet`);
            return;
          }
          lsSampleBufRef.current = [];
          const wav = encodeWAV(bufs, 16000, 2);
          const kb = wav.byteLength / 1024;
          const totalSamples = bufs.reduce((a, b) => a + b.length, 0);
          const durMs = Math.round((totalSamples / 2 / 16000) * 1000);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(wav);
            chunksSent += 1;
            console.log(`[LiveStream] ${id}: 📤 Chunk #${chunksSent} sent — ${kb.toFixed(1)} KB, ~${durMs}ms`);
          } else {
            console.warn(`[LiveStream] ${id}: ⚠️ Cannot send chunk — readyState=${ws.readyState}`);
          }
        }, 1000);
        console.log(`[LiveStream] ${id}: Send interval active (every 1s)`);
      } else {
        console.warn(`[LiveStream] ${id}: ⚠️ No audio streams — WebSocket open but sending no audio`);
      }
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "ping") {
          console.log(`[LiveStream] ${id}: 🏓 Keepalive ping → sending pong`);
          if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "pong" }));
          return;
        }
        if (msg.type === "forensics_chunk_preview") {
          console.log(`[LiveStream] ${id}: 📊 forensics_chunk_preview #${msg.chunk_index} @${msg.elapsed_seconds}s`);
          setLsLatest(prev => ({
            ...prev,
            chunk_index: msg.chunk_index,
            elapsed_seconds: msg.elapsed_seconds,
            spectrogram_analysis: {
              ...prev?.spectrogram_analysis,
              spectrogram_image: msg.spectrogram_image
            }
          } as any));
        }
        if (msg.type === "forensics_final_report") {
          console.log(`[LiveStream] ${id}: 🏆 forensics_final_report received`, msg);
          setLsLatest(msg as any);
          setLsHistory(prev => [...prev.slice(-49), msg as any]);
          sessionEnded = true;
          setLsStatus("ended");
          cleanupLiveStream();
        }
        if (msg.type === "call_ended") {
          console.log(`[LiveStream] ${id}: call_ended received`, msg);
          if (msg.forensics_skipped) {
            setLsLatest(msg as any);
          }
          sessionEnded = true;
          setLsStatus("ended");
          cleanupLiveStream();
        }
      } catch { /* ignore parse errors */ }
    };

    ws.onerror = (ev) => {
      console.error(`[LiveStream] ${id}: ❌ WebSocket ERROR`, ev);
      sessionEnded = true;
      setLsStatus("ended");
      cleanupLiveStream();
    };

    // IMPORTANT: use local sessionEnded flag — reading React state here gives stale values
    ws.onclose = (ev) => {
      console.warn(
        `[LiveStream] ${id}: 🔌 WebSocket CLOSED`,
        `code=${ev.code} reason="${ev.reason}" clean=${ev.wasClean} alreadyEnded=${sessionEnded}`
      );
      if (!sessionEnded) {
        sessionEnded = true;
        setLsStatus("ended");
        cleanupLiveStream();
      }
    };
  };

  // ── Live Stream: disconnect ──────────────────────────────────────────────────
  const disconnectLiveStream = () => {
    console.log(`[LiveStream]: Ending recording — sending END to backend`);
    try { lsWsRef.current?.send("END"); } catch { }
    cleanupLiveStream();
    // Do NOT close the WS yet; wait for the final report or call_ended to arrive.
  };


  // ── Render ───────────────────────────────────────────────────────────────────
  const showLiveDashboard = activeTab === "live" && lsStatus !== "idle";

  return (
    <main className="min-h-screen bg-background relative text-on-background selection:bg-primary selection:text-black pb-20">
      <div className="fixed inset-0 noise-overlay z-50 pointer-events-none" />
      <Navbar />

      <div className="max-w-6xl mx-auto px-6 pt-32 relative z-10">
        {/* Hero */}
        <div className="text-center mb-12">
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="flex justify-center mb-4">
            <span className="text-primary font-headline font-bold uppercase tracking-[0.3em] text-[10px] bg-primary/10 px-3 py-1 rounded-full border border-primary/20">
              AUDIO SPEECH FORENSICS
            </span>
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
            className="text-4xl sm:text-5xl font-extrabold tracking-tight font-headline text-white mb-4">
            Voice <span className="text-primary drop-shadow-[0_0_8px_rgba(49,227,104,0.3)]">Forensics</span>
          </motion.h1>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }}
            className="max-w-2xl mx-auto text-on-surface-variant font-light text-sm sm:text-base">
            {activeTab === "live"
              ? "Stream live call audio through the forensic engine. Voice Clone Detection, Spectrogram Analysis, and Threat Fusion update every 4 seconds."
              : "Submit an audio recording to determine if it is authentic human speech, text-to-speech, or a cloned voice fake."}
          </motion.p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* ── Left column: controls ───────────────────────────────────────── */}
          <div className="lg:col-span-5 space-y-6">
            {/* Tab switcher */}
            <div className="flex border-b border-outline/10 mb-6 font-headline">
              {(["upload", "live"] as const).map(tab => (
                <button key={tab} type="button"
                  onClick={() => { if (lsStatus === "idle" && !loading) { setActiveTab(tab); clearFile(); } }}
                  disabled={lsStatus !== "idle" || loading}
                  className={`flex-1 py-3 text-center border-b-2 text-sm font-semibold transition-all cursor-pointer ${
                    activeTab === tab ? "border-primary text-primary" : "border-transparent text-on-surface-variant hover:text-white disabled:opacity-50"
                  }`}>
                  {tab === "upload" ? "Upload Audio" : "Live Call"}
                </button>
              ))}
            </div>

            {/* ── Upload Tab ───────────────────────────────────────────────── */}
            {activeTab === "upload" && (
              <div
                onDragEnter={handleDrag} onDragOver={handleDrag}
                onDragLeave={handleDrag} onDrop={handleDrop}
                className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all bg-surface/30 backdrop-blur-md flex flex-col items-center justify-center min-h-[300px] group relative ${
                  isDragActive ? "border-primary bg-primary/5 shadow-[0_0_20px_rgba(49,227,104,0.15)]" : "border-outline/20 hover:border-primary/50 hover:bg-surface/50"
                }`}
              >
                <input ref={fileInputRef} type="file" accept=".wav,.mp3,.m4a,.flac" onChange={handleFileChange} className="hidden" />
                {!file ? (
                  <>
                    <div className="p-4 rounded-full bg-surface/80 border border-outline/10 group-hover:border-primary/40 group-hover:text-primary transition-all mb-4 text-on-surface-variant">
                      <Upload className="w-8 h-8 transition-transform group-hover:-translate-y-1 duration-200" />
                    </div>
                    <h3 className="text-white font-headline font-semibold text-lg mb-2">Upload Audio Sample</h3>
                    <p className="text-on-surface-variant text-xs font-light max-w-xs mb-6">
                      Drag and drop your audio file here, or click to browse. Supports WAV, MP3, M4A, FLAC up to 50 MB.
                    </p>
                    <button onClick={() => fileInputRef.current?.click()}
                      className="border border-primary text-primary hover:bg-primary hover:text-black font-semibold text-sm px-6 py-2 rounded-lg transition-all font-headline tracking-wide uppercase active:scale-95 cursor-pointer">
                      Select File
                    </button>
                  </>
                ) : (
                  <>
                    <div className="p-4 rounded-full bg-primary/10 border border-primary/20 text-primary mb-4 animate-bounce">
                      <FileAudio className="w-8 h-8" />
                    </div>
                    <h3 className="text-white font-headline font-semibold text-lg mb-1 truncate max-w-xs">{file.name}</h3>
                    <p className="text-primary font-mono text-xs mb-6">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    <div className="flex gap-4">
                      <button onClick={runAnalysis} disabled={loading}
                        className="bg-primary text-black hover:bg-primary-dark disabled:opacity-50 font-bold text-sm px-6 py-2.5 rounded-lg transition-all font-headline tracking-wide uppercase flex items-center gap-2 active:scale-95 cursor-pointer shadow-lg shadow-primary/20">
                        {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing</> : "Analyze Audio"}
                      </button>
                      <button onClick={clearFile} disabled={loading}
                        className="border border-outline/30 text-on-surface-variant hover:text-white hover:border-white disabled:opacity-50 font-semibold text-sm px-4 py-2.5 rounded-lg transition-all font-headline active:scale-95 cursor-pointer">
                        Remove
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* ── Live Call Tab ─────────────────────────────────────────────── */}
            {activeTab === "live" && (
              <div className="bg-surface/30 backdrop-blur-md border border-outline/10 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[300px] text-center">
                {lsStatus === "idle" && (
                  <>
                    <div className="relative mb-6">
                      <div className="p-5 rounded-full bg-primary/10 border border-primary/20 text-primary">
                        <Radio className="w-8 h-8" />
                      </div>
                    </div>
                    <h3 className="text-white font-headline font-semibold text-xl mb-2">Live Call Forensics</h3>
                    <p className="text-on-surface-variant text-xs font-light max-w-xs mb-2">
                      Shares the same streaming pipeline as Voice Lab.
                    </p>
                    <p className="text-on-surface-variant text-xs font-light max-w-xs mb-8">
                      Opens mic + optional tab capture. Voice Clone Detection, Spectrogram Analysis, and Threat Fusion run every <span className="text-primary font-semibold">4 seconds</span> throughout the call.
                    </p>
                    <button onClick={connectLiveStream}
                      className="bg-primary text-black hover:bg-primary-dark font-bold text-sm px-8 py-3 rounded-lg transition-all font-headline tracking-wide uppercase flex items-center gap-2 active:scale-95 cursor-pointer shadow-lg shadow-primary/20">
                      <Mic className="w-4 h-4" /> Connect &amp; Capture Call Audio
                    </button>
                    <p className="text-on-surface-variant text-[10px] mt-4 font-light">
                      Choose your WhatsApp Web tab and tick "Share tab audio" to capture the caller's voice.
                    </p>
                  </>
                )}

                {lsStatus === "connecting" && (
                  <div className="flex flex-col items-center gap-4">
                    <Loader2 className="w-10 h-10 text-primary animate-spin" />
                    <p className="text-white font-headline font-semibold">Connecting…</p>
                  </div>
                )}

                {(lsStatus === "connected" || lsStatus === "ended") && (
                  <div className="w-full space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs text-on-surface-variant font-mono">
                        <Clock className="w-3.5 h-3.5" /> {fmtDuration(lsDuration)}
                      </div>
                      {lsStatus === "connected" ? (
                        <button onClick={disconnectLiveStream}
                          className="flex items-center gap-1.5 text-xs font-bold text-red-400 border border-red-500/20 px-3 py-1.5 rounded-lg hover:border-red-500/50 transition-all">
                          <PhoneOff className="w-3.5 h-3.5" /> End Call
                        </button>
                      ) : (
                        <button onClick={() => { setLsStatus("idle"); setLsLatest(null); setLsHistory([]); setLsDuration(0); }}
                          className="flex items-center gap-1.5 text-xs font-bold text-primary border border-primary/20 px-3 py-1.5 rounded-lg hover:border-primary/50 transition-all">
                          <Mic className="w-3.5 h-3.5" /> New Session
                        </button>
                      )}
                    </div>
                    {lsCaptureError && (
                      <div className="bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] p-2 rounded-lg flex items-start gap-1.5 text-left">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>{lsCaptureError}</span>
                      </div>
                    )}
                    <div className={`text-[10px] font-mono font-bold px-2 py-1 rounded ${
                      lsStatus === "connected" ? "text-primary" : "text-on-surface-variant"
                    }`}>
                      SESSION · {lsCallId}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error box (upload) */}
            {error && activeTab === "upload" && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="bg-red-500/10 border border-red-500/20 text-red-500 text-xs p-4 rounded-xl flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <div>
                  <p className="font-bold uppercase tracking-wider font-headline text-[10px]">Error Detected</p>
                  <p className="font-light mt-0.5">{error}</p>
                </div>
              </motion.div>
            )}
          </div>

          {/* ── Right column: results / live dashboard ──────────────────────── */}
          <div className="lg:col-span-7">
            <AnimatePresence mode="wait">

              {/* Live dashboard */}
              {showLiveDashboard && (
                <motion.div key="live-dashboard" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                  <LiveDashboard
                    status={lsStatus as "connecting" | "connected" | "ended"}
                    callId={lsCallId}
                    duration={lsDuration}
                    latest={lsLatest}
                    history={lsHistory}
                    captureError={lsCaptureError}
                    onDisconnect={disconnectLiveStream}
                  />
                </motion.div>
              )}

              {/* Upload loading */}
              {loading && activeTab === "upload" && (
                <motion.div key="loading" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                  className="bg-surface/30 backdrop-blur-md border border-outline/10 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[450px]">
                  <Loader2 className="w-12 h-12 text-primary animate-spin mb-6" />
                  <h3 className="text-white font-headline font-semibold text-xl mb-2">Analyzing Voice Signature</h3>
                  <p className="text-on-surface-variant text-sm font-light max-w-sm">
                    Querying neural classification network and calculating physical harmonics (F0 jitter, zero-cross std, MFCC variance) in real-time…
                  </p>
                </motion.div>
              )}

              {/* Upload result */}
              {result && !loading && activeTab === "upload" && (
                <motion.div key="result" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                  <ForensicsResultPanel result={result} />
                </motion.div>
              )}

              {/* Empty state */}
              {!showLiveDashboard && !result && !loading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="border border-outline/10 bg-surface/10 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[450px]">
                  <div className="p-4 rounded-full bg-surface/80 border border-outline/5 text-on-surface-variant mb-4">
                    {activeTab === "live"
                      ? <Radio className="w-8 h-8 text-on-surface-variant" />
                      : <Activity className="w-8 h-8 text-on-surface-variant" />}
                  </div>
                  <h3 className="text-white font-headline font-semibold text-lg mb-2">
                    {activeTab === "live" ? "Live Forensics Ready" : "Awaiting Audio Sample"}
                  </h3>
                  <p className="text-on-surface-variant text-sm font-light max-w-sm">
                    {activeTab === "live"
                      ? <>Click <strong>Connect &amp; Capture Call Audio</strong> to start streaming. Forensic results appear automatically every 4 seconds.</>
                      : "Upload your WAV, MP3, M4A, or FLAC recording. The anti-spoofing fusion engine will execute physical checks and deep network models."}
                  </p>
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </div>
      </div>
    </main>
  );
}
