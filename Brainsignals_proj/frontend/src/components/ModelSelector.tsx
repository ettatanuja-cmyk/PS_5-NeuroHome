import { useEffect, useState } from 'react';
import { Cpu } from 'lucide-react';
import { fetchModels, type ModelOption } from '../api';

interface Props {
  selected: string;
  onChange: (id: string) => void;
}

const FALLBACK: ModelOption[] = [
  { id: 'eeg_tcfnet', name: 'EEG-TCFNet (Base Paper Model)' },
  { id: 'eegnet', name: 'EEGNet (Spatial-Temporal Depthwise)' },
  { id: 'eeg_conformer', name: 'EEG-Conformer (CNN + Self-Attention Transformer)' },
  { id: 'cnn_bilstm', name: 'CNN + BiLSTM (Bidirectional Recurrent)' },
  { id: 'logistic_regression', name: 'Logistic Regression (16-Feature Baseline)' },
];

export default function ModelSelector({ selected, onChange }: Props) {
  const [models, setModels] = useState<ModelOption[]>(FALLBACK);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((j) => {
        // Backend returns both shapes: {models, default} and legacy
        // {available_models, active_default}.
        const list = j.models ?? j.available_models ?? [];
        const def = j.default ?? j.active_default ?? '';
        if (alive && list.length) {
          setModels(list);
          if (!list.some((m) => m.id === selected) && def) {
            onChange(def);
          }
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <label className="flex items-center gap-2 rounded-xl bg-slate-900 p-3 text-xs text-slate-300">
      <Cpu size={16} className="shrink-0 text-sky-400" />
      <span className="font-semibold uppercase tracking-wider text-slate-400">Model</span>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="w-full cursor-pointer rounded-lg border border-slate-700 bg-slate-800 px-2 py-2 text-xs font-medium text-slate-100 outline-none focus:border-sky-400"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
          </option>
        ))}
      </select>
    </label>
  );
}
