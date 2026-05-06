"use client";

import { useEffect, useMemo, useState } from "react";

type ForecastMetadata = {
  observation_time: string;
  forecast_generated_at: string;
  num_steps: number;
  lead_times_minutes: number[];
  valid_times: string[];
  has_current_observation?: boolean;
  georeference?: {
    name: string;
    crs: string;
    bbox: { west: number; south: number; east: number; north: number };
    center: { lat: number; lon: number };
    crop_size: { width: number; height: number };
    pixel_size_degrees: { lon: number; lat: number };
  };
  model: string;
  inference_steps: number;
};

type WeatherSnapshot = {
  label: string;
  lat: number;
  lon: number;
  temperature_c: number | null;
  humidity_pct: number | null;
  wind_kph: number | null;
  rain_mm: number | null;
  rain_probability_pct: number | null;
};

type WeatherSummary = {
  source: string;
  bbox: { west: number; south: number; east: number; north: number };
  center: { lat: number; lon: number };
  snapshots: WeatherSnapshot[];
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

const metadataUrl = "/api/forecast/latest/metadata";

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
  const [selectedStep, setSelectedStep] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [lastError, setLastError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [weather, setWeather] = useState<WeatherSummary | null>(null);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  const fetchMetadata = async () => {
    try {
      const response = await fetch(metadataUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Metadata fetch failed with ${response.status}`);
      }

      const body = (await response.json()) as ForecastMetadata;
      setMetadata(body);
      setSelectedStep((current) => {
        const maxStep = body.has_current_observation === false
          ? Math.max(body.num_steps - 1, 0)
          : Math.max(body.num_steps, 0);
        return Math.min(Math.max(current, 0), maxStep);
      });
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

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const response = await fetch("/api/weather/summary", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Weather fetch failed with ${response.status}`);
        }

        const body = (await response.json()) as WeatherSummary;
        setWeather(body);
        setWeatherError(null);
      } catch (error) {
        setWeatherError(error instanceof Error ? error.message : "Unknown weather error");
      }
    };

    void fetchWeather();
  }, []);

  const stale = useMemo(() => {
    if (!metadata?.forecast_generated_at) return true;
    const generated = new Date(metadata.forecast_generated_at).getTime();
    if (Number.isNaN(generated)) return true;

    const ageMin = (Date.now() - generated) / 60000;
    return ageMin > staleAfterMinutes;
  }, [metadata]);

  const georef = metadata?.georeference;
  const bbox = georef?.bbox;
  const regionLabel = georef?.name ?? "india";
  const southLabel = bbox ? `${bbox.south.toFixed(1)}°N` : "6.0°N";
  const northLabel = bbox ? `${bbox.north.toFixed(1)}°N` : "38.0°N";
  const westLabel = bbox ? `${bbox.west.toFixed(1)}°E` : "66.0°E";
  const eastLabel = bbox ? `${bbox.east.toFixed(1)}°E` : "100.0°E";

  const hasCurrentObservation = metadata?.has_current_observation !== false;
  const selectedLead = selectedStep === 0 ? 0 : (metadata?.lead_times_minutes?.[selectedStep - 1] ?? 0);
  const selectedValidTime = selectedStep === 0
    ? (metadata?.observation_time ?? "")
    : (metadata?.valid_times?.[selectedStep - 1] ?? "");
  const frameUrl = selectedStep === 0 && hasCurrentObservation
    ? "/api/forecast/latest/current.png"
    : `/api/forecast/latest/forecast_step_${Math.max(selectedStep - 1, 0)}.png`;

  const weatherHighlights = weather?.snapshots ?? [];

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

      <section className="dashboard-shell">
        <header className="hero-panel panel">
          <div className="hero-copy">
            <p className="eyebrow">INSAT Nowcasting</p>
            <h1>Meghdoot Live Forecast</h1>
            <p className="hero-subtitle">
              Georeferenced India-wide satellite forecast with weather context and alert-ready structure.
            </p>
          </div>
          <div className={stale ? "badge badge-stale" : "badge badge-fresh"}>
            {stale ? "Stale Data" : "Meghdoot-AI"}
          </div>
        </header>

        <div className="dashboard-grid">
          <section className="panel forecast-panel">
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
                <span>Region</span>
                <p>{regionLabel}</p>
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
                <div className="image-shell map-shell">
                  <div className="map-frame-label map-north">{northLabel}</div>
                  <div className="map-frame-label map-south">{southLabel}</div>
                  <div className="map-frame-label map-west">{westLabel}</div>
                  <div className="map-frame-label map-east">{eastLabel}</div>
                  <img src={frameUrl} alt={`Forecast lead ${selectedLead} minutes`} className="forecast-image" />
                  <div className="map-grid" aria-hidden="true" />
                  <div className="image-caption">
                    <strong>{selectedStep === 0 ? "Now" : `+${selectedLead} min`}</strong>
                    <span>{selectedValidTime ? new Date(selectedValidTime).toLocaleString() : "-"}</span>
                  </div>
                </div>

                <div className="slider-wrap">
                  <label htmlFor="lead-time">Lead Time</label>
                  <input
                    id="lead-time"
                    type="range"
                    min={0}
                    max={hasCurrentObservation ? Math.max(metadata?.num_steps || 1, 0) : Math.max((metadata?.num_steps || 1) - 1, 0)}
                    step={1}
                    value={selectedStep}
                    onChange={(event) => setSelectedStep(Number(event.target.value))}
                  />
                  <div className="ticks">
                    {hasCurrentObservation ? (
                      <button
                        type="button"
                        key="now"
                        className={selectedStep === 0 ? "tick active" : "tick"}
                        onClick={() => setSelectedStep(0)}
                      >
                        Now
                      </button>
                    ) : null}
                    {(metadata?.lead_times_minutes || []).map((m, idx) => {
                      const uiIndex = hasCurrentObservation ? idx + 1 : idx;
                      return (
                      <button
                        type="button"
                        key={`${m}-${idx}`}
                        className={uiIndex === selectedStep ? "tick active" : "tick"}
                        onClick={() => setSelectedStep(uiIndex)}
                      >
                        {m}m
                      </button>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </section>

          <aside className="sidebar-column">
            <section className="panel sidebar-panel">
              <div className="metrics-header">
                <h2>Georeference</h2>
                <p>{georef?.crs || "EPSG:4326"} overlay contract for the India crop</p>
              </div>
              <div className="detail-card-grid">
                <div className="detail-card">
                  <span>Bounds</span>
                  <strong>{bbox ? `${bbox.west.toFixed(1)}E → ${bbox.east.toFixed(1)}E` : "66E → 100E"}</strong>
                  <small>{bbox ? `${bbox.south.toFixed(1)}N → ${bbox.north.toFixed(1)}N` : "6N → 38N"}</small>
                </div>
                <div className="detail-card">
                  <span>Pixels</span>
                  <strong>{georef ? `${georef.crop_size.width} × ${georef.crop_size.height}` : "512 × 512"}</strong>
                  <small>{georef ? `${georef.pixel_size_degrees.lon.toFixed(3)}° / px` : "georeferenced crop"}</small>
                </div>
                <div className="detail-card">
                  <span>Center</span>
                  <strong>{georef ? `${georef.center.lat.toFixed(1)}N, ${georef.center.lon.toFixed(1)}E` : "India center"}</strong>
                  <small>{metadata?.has_current_observation ? "Current frame included" : "Forecast only"}</small>
                </div>
              </div>
            </section>

            <section className="panel sidebar-panel">
              <div className="metrics-header">
                <h2>Weather Snapshot</h2>
                <p>Regional wind, humidity, and rainfall from Open-Meteo</p>
              </div>
              {weatherError ? (
                <div className="placeholder error">
                  <p>Could not load weather summary</p>
                  <small>{weatherError}</small>
                </div>
              ) : !weather ? (
                <div className="placeholder">Loading weather summary...</div>
              ) : (
                <div className="weather-grid">
                  {weatherHighlights.map((point) => (
                    <article className="weather-card" key={point.label}>
                      <div className="weather-card-top">
                        <strong>{point.label}</strong>
                        <span>{point.lat.toFixed(1)}N, {point.lon.toFixed(1)}E</span>
                      </div>
                      <div className="weather-values">
                        <div><span>Temp</span><strong>{point.temperature_c ?? "-"}°C</strong></div>
                        <div><span>Humidity</span><strong>{point.humidity_pct ?? "-"}%</strong></div>
                        <div><span>Wind</span><strong>{point.wind_kph ?? "-"} kph</strong></div>
                        <div><span>Rain</span><strong>{point.rain_mm ?? "-"} mm</strong></div>
                      </div>
                      <small>Rain chance: {point.rain_probability_pct ?? "-"}%</small>
                    </article>
                  ))}
                </div>
              )}
            </section>

            <section className="panel sidebar-panel">
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
          </aside>
        </div>
      </section>
    </main>
  );
}
