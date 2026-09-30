import { useEffect, useState } from 'react';

interface Health {
  status: string;
  uptime_seconds?: number;
  checkpoint_loaded?: boolean;
}

function fmtUptime(s?: number) {
  if (s == null) return '—';
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export default function StatusPill({ intervalMs = 5000 }: { intervalMs?: number }) {
  const [online, setOnline] = useState<boolean | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const res = await fetch('/api/health', { cache: 'no-store' });
        if (!res.ok) throw new Error();
        const h = (await res.json()) as Health;
        if (alive) {
          setOnline(true);
          setHealth(h);
        }
      } catch {
        if (alive) setOnline(false);
      }
    };
    check();
    const id = setInterval(check, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  if (online === null) {
    return (
      <span className="inline-flex items-center gap-2 rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-400">
        <span className="h-2 w-2 animate-pulse rounded-full bg-slate-500" /> Checking…
      </span>
    );
  }
  return (
    <span
      title={online ? `Backend uptime: ${fmtUptime(health?.uptime_seconds)}` : 'Backend unreachable — start run_backend.bat'}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
        online ? 'bg-emerald-600/20 text-emerald-300' : 'bg-red-600/20 text-red-300'
      }`}
    >
      <span className="relative flex h-2 w-2">
        {online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${online ? 'bg-emerald-400' : 'bg-red-500'}`} />
      </span>
      {online ? `System ON · up ${fmtUptime(health?.uptime_seconds)}` : 'System OFF'}
    </span>
  );
}
