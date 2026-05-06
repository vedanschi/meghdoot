import { NextResponse } from "next/server";

type ForecastMetadata = {
  georeference?: {
    center?: { lat: number; lon: number };
    bbox?: { west: number; south: number; east: number; north: number };
  };
};

type WeatherPoint = {
  label: string;
  lat: number;
  lon: number;
  temperature_c: number | null;
  humidity_pct: number | null;
  wind_kph: number | null;
  rain_mm: number | null;
  rain_probability_pct: number | null;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

async function fetchMetadata(baseUrl: string): Promise<ForecastMetadata | null> {
  try {
    const response = await fetch(`${baseUrl}/api/forecast/latest/metadata`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as ForecastMetadata;
  } catch {
    return null;
  }
}

async function fetchOpenMeteo(lat: number, lon: number): Promise<Partial<WeatherPoint>> {
  const url = new URL("https://api.open-meteo.com/v1/forecast");
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("current", "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m");
  url.searchParams.set("hourly", "precipitation_probability,precipitation");
  url.searchParams.set("timezone", "auto");

  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Open-Meteo fetch failed with ${response.status}`);
  }

  const payload = await response.json() as {
    current?: {
      temperature_2m?: number;
      relative_humidity_2m?: number;
      precipitation?: number;
      wind_speed_10m?: number;
    };
    hourly?: {
      time?: string[];
      precipitation_probability?: number[];
      precipitation?: number[];
    };
  };

  const hourlyIndex = payload.hourly?.time?.findIndex((entry) => entry?.startsWith(new Date().toISOString().slice(0, 13))) ?? -1;
  const rainProbability = hourlyIndex >= 0 ? payload.hourly?.precipitation_probability?.[hourlyIndex] ?? null : null;
  const rainMm = hourlyIndex >= 0 ? payload.hourly?.precipitation?.[hourlyIndex] ?? null : null;

  return {
    temperature_c: payload.current?.temperature_2m ?? null,
    humidity_pct: payload.current?.relative_humidity_2m ?? null,
    wind_kph: payload.current?.wind_speed_10m ?? null,
    rain_mm: payload.current?.precipitation ?? rainMm,
    rain_probability_pct: rainProbability,
  };
}

export async function GET(request: Request) {
  const baseUrl = new URL(request.url).origin;
  const metadata = await fetchMetadata(baseUrl);
  const bbox = metadata?.georeference?.bbox ?? { west: 66.0, south: 6.0, east: 100.0, north: 38.0 };
  const center = metadata?.georeference?.center ?? {
    lat: (bbox.north + bbox.south) / 2,
    lon: (bbox.east + bbox.west) / 2,
  };

  const points = [
    { label: "North", lat: clamp(bbox.north - 3, bbox.south, bbox.north), lon: center.lon },
    { label: "Center", lat: center.lat, lon: center.lon },
    { label: "West", lat: center.lat, lon: clamp(bbox.west + 3, bbox.west, bbox.east) },
    { label: "East", lat: center.lat, lon: clamp(bbox.east - 3, bbox.west, bbox.east) },
    { label: "South", lat: clamp(bbox.south + 3, bbox.south, bbox.north), lon: center.lon },
  ];

  try {
    const snapshots = await Promise.all(
      points.map(async (point) => ({
        ...point,
        ...(await fetchOpenMeteo(point.lat, point.lon)),
      })),
    );

    return NextResponse.json({
      source: "open-meteo",
      bbox,
      center,
      snapshots,
    }, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown weather summary error";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}