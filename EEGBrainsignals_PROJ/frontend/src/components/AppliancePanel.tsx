import { Users, Layers, Ratio, MonitorSmartphone } from 'lucide-react';
import type { ApplianceInfo } from '../api';
import { ApplianceIcon } from './ApplianceIcon';

interface Props {
  catalog: ApplianceInfo[];
  selected: string | null;
  onSelect: (name: string) => void;
  highlight?: string | null;
}

export default function AppliancePanel({ catalog, selected, onSelect, highlight }: Props) {
  const active = catalog.find((a) => a.name === selected) ?? null;
  return (
    <section className="rounded-2xl bg-slate-900 p-5">
      <h2 className="text-sm font-semibold text-slate-300">Dataset applications — choose one to see its commands</h2>
      <p className="mt-1 text-[11px] text-slate-500">
        Your choice is also sent with the prediction so the decoded command is labelled with the right mapping.
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {catalog.map((a) => {
          const isSel = a.name === selected;
          const isHi = a.name === highlight && !isSel;
          return (
            <button
              key={a.name}
              onClick={() => onSelect(a.name)}
              className={`flex flex-col items-center gap-1 rounded-xl border p-3 text-xs transition ${
                isSel
                  ? 'border-sky-400 bg-sky-950/50 text-sky-200'
                  : isHi
                    ? 'border-emerald-500 bg-emerald-950/40 text-emerald-200'
                    : 'border-slate-700 bg-slate-800/60 text-slate-300 hover:border-slate-500'
              }`}
            >
              <ApplianceIcon name={a.name} size={26} />
              <span className="font-medium">{a.name}</span>
              <span className="text-[10px] opacity-70">{a.subjects} subjects</span>
            </button>
          );
        })}
      </div>

      {active ? (
        <div className="mt-4 rounded-xl border border-slate-700 bg-slate-800/50 p-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
            <span className="inline-flex items-center gap-1"><Layers size={13} /> {active.paradigm}</span>
            <span className="inline-flex items-center gap-1"><Users size={13} /> {active.subjects} subjects</span>
            <span className="inline-flex items-center gap-1"><Ratio size={13} /> target ratio {active.target_ratio}</span>
            <span className="inline-flex items-center gap-1"><MonitorSmartphone size={13} /> {active.environment}</span>
          </div>
          <p className="mt-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Commands in dataset ({active.commands.length})
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {active.commands.map((c) => (
              <span
                key={c.stimulus_id}
                className="inline-flex items-center gap-1.5 rounded-full bg-slate-700/70 px-3 py-1 text-xs text-slate-200"
              >
                <span className="rounded-full bg-sky-600 px-1.5 font-mono text-[10px]">{c.stimulus_id}</span>
                {c.label}
              </span>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            {active.note} Folder: <span className="font-mono">{active.folder}/</span> · {active.repetitions_per_block} flickers per
            stimulus per block.
          </p>
        </div>
      ) : (
        <p className="mt-3 text-xs text-slate-500">Select an application above to list the control commands recorded for it.</p>
      )}
    </section>
  );
}
