const BAR_COLORS: Record<string, string> = {
  'Air Conditioner': 'bg-sky-500',
  'Bluetooth Speaker': 'bg-violet-500',
  'Door Lock': 'bg-emerald-500',
  'Electric Light': 'bg-amber-400',
  TV: 'bg-rose-500'
};

export default function ConfidenceBars({ scores }: { scores: Record<string, number> }) {
  const rows = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-2">
      {rows.map(([label, p]) => (
        <div key={label}>
          <div className="flex justify-between text-xs text-slate-300">
            <span>{label}</span>
            <span className="font-mono">{(p * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded bg-slate-800">
            <div
              className={`h-2 rounded ${BAR_COLORS[label] ?? 'bg-slate-400'}`}
              style={{ width: `${Math.max(p * 100, 2)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
