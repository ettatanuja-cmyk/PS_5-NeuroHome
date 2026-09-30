import { useCallback, useEffect, useState } from 'react';
import { UploadCloud, FileUp, BrainCircuit, Activity, Zap, AlertCircle, Loader2 } from 'lucide-react';
import { predictFile, fetchAppliances, type PredictResponse, type ApplianceInfo } from './api';
import ModelSelector from './components/ModelSelector';
import SignalChart from './components/SignalChart';
import ConfidenceBars from './components/ConfidenceBars';
import StatusPill from './components/StatusPill';
import AppliancePanel from './components/AppliancePanel';
import { ApplianceIcon } from './components/ApplianceIcon';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [catalog, setCatalog] = useState<ApplianceInfo[]>([]);
  const [selectedApp, setSelectedApp] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('eeg_tcfnet');

  useEffect(() => {
    fetchAppliances().then(setCatalog).catch(() => {});
  }, []);

  const pick = useCallback((f: File | undefined) => {
    setError('');
    setResult(null);
    if (!f) return;
    if (!/\.mat$|\.zip$/i.test(f.name)) {
      setError('Please upload a .mat file or a .zip containing .mat files.');
      return;
    }
    setFile(f);
  }, []);

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const r = await predictFile(file, 'mean', selectedApp ?? '', selectedModel);
      setResult(r);
    } catch (e: any) {
      setError(e?.message ?? 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-6 py-4">
          <BrainCircuit className="text-sky-400" size={28} />
          <div className="flex-1">
            <h1 className="text-lg font-semibold">BCI Home Appliance Control · Multi-Model</h1>
            <p className="text-xs text-slate-400">
              Upload EEG <span className="font-mono">.mat</span> trials (sig_vec / cal_sig / Data struct) or a{' '}
              <span className="font-mono">.zip</span> — 120 Hz resample · 1–15 Hz Butterworth · pick EEGNet / Conformer / CNN-BiLSTM / LogReg
            </p>
          </div>
          <StatusPill />
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        {catalog.length > 0 && (
          <AppliancePanel
            catalog={catalog}
            selected={selectedApp}
            onSelect={setSelectedApp}
            highlight={result?.predicted_appliance ?? null}
          />
        )}
        <div className="grid gap-6 lg:grid-cols-5">
        {/* Upload panel */}
        <section className="lg:col-span-2">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              pick(e.dataTransfer.files?.[0]);
            }}
            className={`rounded-2xl border-2 border-dashed p-8 text-center transition ${
              drag ? 'border-sky-400 bg-sky-950/40' : 'border-slate-700 bg-slate-900'
            }`}
          >
            <UploadCloud className="mx-auto text-slate-400" size={40} />
            <p className="mt-3 text-sm text-slate-300">Drag &amp; drop your .mat / .zip here</p>
            <label className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500">
              <FileUp size={16} /> Choose file
              <input
                type="file"
                accept=".mat,.zip"
                className="hidden"
                onChange={(e) => pick(e.target.files?.[0])}
              />
            </label>
            {file && <p className="mt-3 font-mono text-xs text-sky-300">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
          </div>

          <button
            onClick={run}
            disabled={!file || loading}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-3 font-semibold disabled:opacity-40 hover:bg-emerald-500"
          >
            {loading ? <Loader2 className="animate-spin" size={18} /> : <Zap size={18} />}
            {loading ? `Running ${selectedModel}…` : 'Predict appliance'}
          </button>

          <div className="mt-4">
            <ModelSelector selected={selectedModel} onChange={setSelectedModel} />
          </div>

          {error && (
            <p className="mt-3 flex items-start gap-2 rounded-lg bg-red-950/60 p-3 text-sm text-red-300">
              <AlertCircle size={16} className="mt-0.5 shrink-0" /> {error}
            </p>
          )}

          <div className="mt-4 rounded-xl bg-slate-900 p-4 text-xs leading-5 text-slate-400">
            <p className="font-semibold text-slate-300">Pipeline</p>
            <ol className="list-decimal pl-4">
              <li>scipy.io.loadmat → sig_vec / cal_sig / Data.signal</li>
              <li>Resample 500 Hz → 120 Hz, 6th-order 1–15 Hz bandpass</li>
              <li>Z-score + outlier clip → selected model (EEGNet / Conformer / CNN-BiLSTM / LogReg) → Softmax</li>
            </ol>
          </div>
        </section>

        {/* Results panel */}
        <section className="lg:col-span-3">
          {!result ? (
            <div className="flex h-full min-h-64 flex-col items-center justify-center rounded-2xl bg-slate-900 p-10 text-center text-slate-500">
              <Activity size={36} />
              <p className="mt-3 text-sm">No prediction yet — upload a trial file and press Predict.</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center gap-4 rounded-2xl bg-gradient-to-r from-sky-950 to-slate-900 p-5">
                <div className="rounded-xl bg-sky-600/20 p-3 text-sky-300">
                  <ApplianceIcon name={result.predicted_appliance} />
                </div>
                <div className="flex-1">
                  <p className="text-xs uppercase tracking-wider text-slate-400">Predicted appliance</p>
                  <p className="text-2xl font-bold">{result.predicted_appliance}</p>
                  <p className="text-xs text-slate-400">
                    class_id {result.class_id} · top confidence{' '}
                    {(result.confidence_scores[result.predicted_appliance] * 100).toFixed(1)}%
                    {result.files_processed && result.files_processed > 1 ? ` · averaged over ${result.files_processed} files` : ''}
                    {result.model_used ? ` · model: ${result.model_used}` : ''}
                    {result.model_source ? ` (${result.model_source})` : ''}
                  </p>
                  {result.predicted_command ? (
                    <p className="mt-1 inline-flex items-center gap-1.5 rounded-full bg-emerald-600/20 px-3 py-1 text-sm font-semibold text-emerald-300">
                      <Zap size={14} /> Command: {result.predicted_command.label}
                      <span className="font-mono text-xs opacity-80">
                        (stimulus {result.predicted_command.stimulus_id} · {(result.predicted_command.confidence * 100).toFixed(1)}%)
                      </span>
                    </p>
                  ) : (
                    <p className="mt-1 text-xs text-slate-500">Command: n/a — file has no trigger channel (e.g. calibration signal).</p>
                  )}
                </div>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    result.p300_detected ? 'bg-emerald-600/30 text-emerald-300' : 'bg-slate-700 text-slate-300'
                  }`}
                >
                  {result.p300_detected ? `P300 ✓ ${result.p300_score ?? ''}` : 'P300 ✗'}
                </span>
              </div>

              <div className="rounded-2xl bg-slate-900 p-5">
                <h2 className="mb-3 text-sm font-semibold text-slate-300">Confidence scores</h2>
                <ConfidenceBars scores={result.confidence_scores} />
              </div>

              {result.predicted_command && (
                <div className="rounded-2xl bg-slate-900 p-5">
                  <h2 className="mb-1 text-sm font-semibold text-slate-300">
                    Command decoding — ERP target detection
                  </h2>
                  <p className="mb-3 text-[11px] text-slate-500">
                    {result.predicted_command.method} · stimulus labels mapped via{' '}
                    <span className="font-semibold text-slate-300">{result.predicted_command.context_appliance}</span> ·
                    decoded from <span className="font-mono">{result.predicted_command.file}</span>
                  </p>
                  <ConfidenceBars scores={result.predicted_command.command_scores} />
                </div>
              )}

              {result.appliance_info && (
                <div className="rounded-2xl bg-slate-900 p-5">
                  <h2 className="mb-1 text-sm font-semibold text-slate-300">
                    Dataset commands for {result.predicted_appliance}
                  </h2>
                  <p className="mb-2 text-[11px] text-slate-500">
                    {result.appliance_info.paradigm} · {result.appliance_info.subjects} subjects · target ratio{' '}
                    {result.appliance_info.target_ratio}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {result.appliance_info.commands.map((c) => (
                      <span
                        key={c.stimulus_id}
                        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs ${
                          result.predicted_command?.stimulus_id === c.stimulus_id
                            ? 'bg-emerald-600/30 text-emerald-200 ring-1 ring-emerald-400'
                            : 'bg-slate-700/70 text-slate-200'
                        }`}
                      >
                        <span className="rounded-full bg-emerald-600 px-1.5 font-mono text-[10px]">{c.stimulus_id}</span>
                        {c.label}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="rounded-2xl bg-slate-900 p-5">
                <h2 className="mb-1 text-sm font-semibold text-slate-300">Preprocessed EEG preview (120 Hz, 1–15 Hz)</h2>
                <p className="mb-3 font-mono text-[11px] text-slate-500">
                  {result.meta?.source_key} · {result.meta?.channels} ch × {result.meta?.samples} samples · file: {result.file ?? result.files?.[0] ?? '—'}
                </p>
                <SignalChart plot={result.signal_plot} fallback={result.signal_preview} />
              </div>

              {result.files && result.files.length > 1 && (
                <div className="rounded-2xl bg-slate-900 p-5 text-xs text-slate-400">
                  <h2 className="mb-2 text-sm font-semibold text-slate-300">Files in archive</h2>
                  <ul className="max-h-32 list-disc overflow-auto pl-5 font-mono">
                    {result.files.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
        </div>
      </main>
    </div>
  );
}
