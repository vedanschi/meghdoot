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

const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
const prefix = process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest";
const refreshMs = Number(process.env.NEXT_PUBLIC_REFRESH_MS || "60000");
const staleAfterMinutes = Number(process.env.NEXT_PUBLIC_STALE_AFTER_MINUTES || "90");

const metadataUrl = `https://storage.googleapis.com/${bucket}/${prefix}/metadata.json`;

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

  const stale = useMemo(() => {
    if (!metadata?.forecast_generated_at) return true;
    const generated = new Date(metadata.forecast_generated_at).getTime();
    if (Number.isNaN(generated)) return true;

    const ageMin = (Date.now() - generated) / 60000;
    return ageMin > staleAfterMinutes;
  }, [metadata]);

  const selectedLead = metadata?.lead_times_minutes?.[selectedStep] ?? 0;
  const selectedValidTime = metadata?.valid_times?.[selectedStep] ?? "";
  const frameUrl = `https://storage.googleapis.com/${bucket}/${prefix}/forecast_step_${selectedStep}.png`;

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
          </>
        )}
      </section>
    </main>
  );
}
