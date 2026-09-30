import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import type { SignalPlot } from '../api';

const COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#fbbf24'];

export default function SignalChart({ plot, fallback }: { plot?: SignalPlot; fallback?: number[] }) {
  let data: { t: number; [k: string]: number }[] = [];
  let channels: string[] = [];
  if (plot && plot.series?.length) {
    channels = plot.series.map((s) => s.channel);
    const n = plot.series[0].values.length;
    data = Array.from({ length: n }, (_, i) => {
      const row: { t: number; [k: string]: number } = { t: plot.time[i] ?? i };
      plot.series.forEach((s) => {
        row[s.channel] = s.values[i];
      });
      return row;
    });
  } else if (fallback?.length) {
    channels = ['Ch1'];
    data = fallback.map((v, i) => ({ t: +(i * 0.0083).toFixed(3), Ch1: v }));
  }
  if (!data.length) return <p className="text-sm text-slate-400">No signal preview available.</p>;
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: 'time (s)', fontSize: 10, fill: '#64748b' }} />
          <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} width={48} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          {channels.map((c, i) => (
            <Line key={c} type="monotone" dataKey={c} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.5} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
