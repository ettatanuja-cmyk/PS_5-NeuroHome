export interface SignalPlot {
  channels: string[];
  time: number[];
  series: { channel: string; values: number[] }[];
  flat: number[];
}

export interface ApplianceCommand {
  stimulus_id: number;
  label: string;
}

export interface ApplianceInfo {
  name: string;
  class_id: number;
  folder: string;
  subjects: number;
  paradigm: string;
  environment: string;
  stimuli: number;
  target_ratio: string;
  repetitions_per_block: number;
  commands: ApplianceCommand[];
  note: string;
}

export interface ModelOption {
  id: string;
  name: string;
}

export async function fetchModels(): Promise<{
  models: ModelOption[];
  default: string;
  available_models: ModelOption[];
  active_default: string;
}> {
  const res = await fetch('/api/models', { cache: 'no-store' });
  if (!res.ok) throw new Error(`Model catalog request failed (${res.status})`);
  const j = await res.json();
  // Normalize: backend sends both {models, default} and legacy
  // {available_models, active_default}.
  const models: ModelOption[] = j.models ?? j.available_models ?? [];
  const defaultId: string = j.default ?? j.active_default ?? 'eeg_tcfnet';
  return {
    models,
    default: defaultId,
    available_models: models,
    active_default: defaultId,
  };
}

export async function fetchAppliances(): Promise<ApplianceInfo[]> {
  const res = await fetch('/api/appliances', { cache: 'no-store' });
  if (!res.ok) throw new Error(`Catalog request failed (${res.status})`);
  const j = await res.json();
  return j.appliances as ApplianceInfo[];
}

export interface PredictedCommand {
  stimulus_id: number;
  label: string;
  confidence: number;
  command_scores: Record<string, number>;
  erp_scores: Record<string, number>;
  repetitions: Record<string, number>;
  method: string;
  file: string;
  context_appliance: string;
}
export interface PredictResponse {
  predicted_appliance: string;
  class_id: number;
  confidence_scores: Record<string, number>;
  p300_detected: boolean;
  p300_score?: number;
  signal_preview: number[];
  signal_plot?: SignalPlot;
  meta?: Record<string, any>;
  file?: string;
  files_processed?: number;
  files?: string[];
  used_fallback?: boolean;
  appliance_info?: ApplianceInfo;
  predicted_command?: PredictedCommand | null;
  model_source?: string;
  model_used?: string;
}

export async function predictFile(
  file: File,
  aggregate = 'first',
  appliance = '',
  model_name = 'eeg_tcfnet',
): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append('file', file);
  const q = new URLSearchParams({ aggregate, appliance, model_name });
  const res = await fetch(`/api/predict?${q.toString()}`, { method: 'POST', body: fd });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `Request failed (${res.status})`);
  }
  return res.json();
}
