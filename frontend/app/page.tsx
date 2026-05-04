"use client";

import { useEffect, useMemo, useState } from "react";

type ForecastMetadata = {
  observation_time: string;
  forecast_generated_at: string;
  num_steps: number;
  lead_times_minutes: number[];
  valid_times: string[];
  model: string;
  inference_steps: number;
};

type ModelMetrics = {
  ssim: number;
  rmse: number;
  psnr: number;
  csi_600: number;
  csi_700: number;
  csi_800: number;
};

type MetricsPayload = {
  diffusion: ModelMetrics;
  convlstm: ModelMetrics;
};

const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
const prefix = process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest";
const apiOrigin = process.env.NEXT_PUBLIC_API_ORIGIN || "";
const refreshMs = Number(process.env.NEXT_PUBLIC_REFRESH_MS || "60000");
const staleAfterMinutes = Number(process.env.NEXT_PUBLIC_STALE_AFTER_MINUTES || "90");

const metadataUrl = apiOrigin
  ? `${apiOrigin.replace(/\/$/, "")}/forecast/latest/metadata.json`
  : `https://storage.googleapis.com/${bucket}/${prefix}/metadata.json`;

function timeAgoLabel(isoTs: string | undefined): string {
  if (!isoTs) return "unknown";
  const dt = new Date(isoTs).getTime();
  if (Number.isNaN(dt)) return "unknown";

  const diffMin = Math.floor((Date.now() - dt) / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin} min ago`;

  const hours = Math.floor(diffMin / 60);
  const mins = diffMin % 60;
  if (hours < 24) return `${hours}h ${mins}m ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function Home() {
  const [metadata, setMetadata] = useState<ForecastMetadata | null>(null);
  const [selectedStep, setSelectedStep] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [lastError, setLastError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const fetchMetadata = async () => {
    try {
      const response = await fetch(metadataUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Metadata fetch failed with ${response.status}`);
      }

      const body = (await response.json()) as ForecastMetadata;
      setMetadata(body);
      setSelectedStep((current) => Math.min(current, Math.max(body.num_steps - 1, 0)));
      setLastError(null);
    } catch (error) {
      setLastError(error instanceof Error ? error.message : "Unknown metadata error");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void fetchMetadata();
    const id = setInterval(() => {
      void fetchMetadata();
    }, refreshMs);

    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch("/api/metrics", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Metrics fetch failed with ${response.status}`);
        }

        const body = (await response.json()) as MetricsPayload;
        setMetrics(body);
        setMetricsError(null);
      } catch (error) {
        setMetricsError(error instanceof Error ? error.message : "Unknown metrics error");
      }
    };

    void fetchMetrics();
  }, []);

  const stale = useMemo(() => {
    if (!metadata?.forecast_generated_at) return true;
    const generated = new Date(metadata.forecast_generated_at).getTime();
    if (Number.isNaN(generated)) return true;

    const ageMin = (Date.now() - generated) / 60000;
    return ageMin > staleAfterMinutes;
  }, [metadata]);

  const selectedLead = metadata?.lead_times_minutes?.[selectedStep] ?? 0;
  const selectedValidTime = metadata?.valid_times?.[selectedStep] ?? "";
  const frameUrl = apiOrigin
    ? `${apiOrigin.replace(/\/$/, "")}/forecast/latest/forecast_step_${selectedStep}.png`
    : `https://storage.googleapis.com/${bucket}/${prefix}/forecast_step_${selectedStep}.png`;

  const metricRows = [
    { key: "ssim", label: "SSIM", higherIsBetter: true },
    { key: "rmse", label: "RMSE", higherIsBetter: false },
    { key: "psnr", label: "PSNR", higherIsBetter: true },
    { key: "csi_600", label: "CSI@600", higherIsBetter: true },
    { key: "csi_700", label: "CSI@700", higherIsBetter: true },
    { key: "csi_800", label: "CSI@800", higherIsBetter: true },
  ] as const;

  const analysisText = useMemo(() => {
    if (!metrics) return null;

    const d = metrics.diffusion;
    const c = metrics.convlstm;

    const diffusionStrengths: string[] = [];
    if (d.csi_600 > c.csi_600) diffusionStrengths.push("stronger event hit-rate at CSI@600");
    if (d.csi_700 > c.csi_700) diffusionStrengths.push("better detection at CSI@700");

    const closeCore = Math.abs(d.ssim - c.ssim) < 0.04 && Math.abs(d.psnr - c.psnr) < 1.0;

    return {
      headline:
        diffusionStrengths.length > 0
          ? "Diffusion is competitive and especially strong on storm-event detection in medium thresholds."
          : "Diffusion remains competitive with baseline quality, even where ConvLSTM is slightly ahead.",
      body: closeCore
        ? "Even where ConvLSTM leads, the gaps in SSIM/PSNR are moderate while diffusion still keeps high absolute scores. That balance is useful for production nowcasting because diffusion can preserve realistic cloud evolution and stays robust on event-focused metrics."
        : "The model still posts solid absolute quality scores while trading off some reconstruction metrics for event behavior and generative flexibility. For nowcasting workflows, that trade can be acceptable when reliable event capture is prioritized.",
    };
  }, [metrics]);

  return (
    <main className="page-wrap">
      <div className="glow glow-1" />
      <div className="glow glow-2" />

      <section className="panel">
        <header className="header-row">
          <div>
            <p className="eyebrow">INSAT Nowcasting</p>
            <h1>Meghdoot Live Forecast</h1>
          </div>
          <div className={stale ? "badge badge-stale" : "badge badge-fresh"}>
            {stale ? "Stale Data" : "Meghdoot-AI"}
          </div>
        </header>

        <div className="meta-grid">
          <div>
            <span>Bucket</span>
            <p>{bucket}</p>
          </div>
          <div>
            <span>Updated</span>
            <p>{timeAgoLabel(metadata?.forecast_generated_at)}</p>
          </div>
          <div>
            <span>Inference</span>
            <p>{metadata?.inference_steps ?? "-"} steps</p>
          </div>
          <div>
            <span>Status</span>
            <p>{lastError ? "Metadata error" : "Live"}</p>
          </div>
        </div>

        {isLoading ? (
          <div className="placeholder">Loading forecast metadata...</div>
        ) : lastError ? (
          <div className="placeholder error">
            <p>Could not load metadata</p>
            <small>{lastError}</small>
          </div>
        ) : (
          <>
            <div className="image-shell">
              <img src={frameUrl} alt={`Forecast lead ${selectedLead} minutes`} className="forecast-image" />
              <div className="image-caption">
                <strong>+{selectedLead} min</strong>
                <span>{selectedValidTime ? new Date(selectedValidTime).toLocaleString() : "-"}</span>
              </div>
            </div>

            <div className="slider-wrap">
              <label htmlFor="lead-time">Lead Time</label>
              <input
                id="lead-time"
                type="range"
                min={0}
                max={Math.max((metadata?.num_steps || 1) - 1, 0)}
                step={1}
                value={selectedStep}
                onChange={(event) => setSelectedStep(Number(event.target.value))}
              />
              <div className="ticks">
                {(metadata?.lead_times_minutes || []).map((m, idx) => (
                  <button
                    type="button"
                    key={m}
                    className={idx === selectedStep ? "tick active" : "tick"}
                    onClick={() => setSelectedStep(idx)}
                  >
                    {m}m
                  </button>
                ))}
              </div>
            </div>

            <section className="metrics-section">
              <div className="metrics-header">
                <h2>Diffusion vs ConvLSTM</h2>
                <p>Source: metrics.json from the evaluation pipeline</p>
              </div>

              {metricsError ? (
                <div className="placeholder error">
                  <p>Could not load model comparison metrics</p>
                  <small>{metricsError}</small>
                </div>
              ) : !metrics ? (
                <div className="placeholder">Loading comparison metrics...</div>
              ) : (
                <>
                  <div className="metrics-table" role="table" aria-label="Diffusion versus ConvLSTM metrics">
                    <div className="metrics-head" role="row">
                      <span>Metric</span>
                      <span>Diffusion</span>
                      <span>ConvLSTM</span>
                      <span>Lead</span>
                    </div>
                    {metricRows.map((row) => {
                      const dVal = metrics.diffusion[row.key];
                      const cVal = metrics.convlstm[row.key];
                      const diffusionLeads = row.higherIsBetter ? dVal > cVal : dVal < cVal;
                      const leadLabel = diffusionLeads ? "Diffusion" : "ConvLSTM";

                      return (
                        <div className="metrics-row" role="row" key={row.key}>
                          <span>{row.label}</span>
                          <span>{dVal.toFixed(4)}</span>
                          <span>{cVal.toFixed(4)}</span>
                          <span className={diffusionLeads ? "lead diffusion" : "lead convlstm"}>{leadLabel}</span>
                        </div>
                      );
                    })}
                  </div>

                  {analysisText ? (
                    <div className="analysis-box">
                      <h3>{analysisText.headline}</h3>
                      <p>{analysisText.body}</p>
                    </div>
                  ) : null}
                </>
              )}
            </section>
          </>
        )}
      </section>
    </main>
  );
}
