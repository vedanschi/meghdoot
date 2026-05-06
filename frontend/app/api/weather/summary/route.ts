import { NextResponse } from "next/server";

type ForecastMetadata = {
  georeference?: {
    center?: { lat: number; lon: number };
    bbox?: { west: number; south: number; east: number; north: number };
  };
};

type WeatherCity = {
  id: string;
  label: string;
  lat: number;
  lon: number;
};

type WeatherPoint = {
  id: string;
  label: string;
  lat: number;
  lon: number;
  temperature_c: number | null;
  humidity_pct: number | null;
  wind_kph: number | null;
  rain_mm: number | null;
  rain_probability_pct: number | null;
};

const INDIAN_CITIES: WeatherCity[] = [
  { id: "delhi", label: "Delhi", lat: 28.6139, lon: 77.2090 },
  { id: "mumbai", label: "Mumbai", lat: 19.0760, lon: 72.8777 },
  { id: "kolkata", label: "Kolkata", lat: 22.5726, lon: 88.3639 },
  { id: "chennai", label: "Chennai", lat: 13.0827, lon: 80.2707 },
  { id: "bengaluru", label: "Bengaluru", lat: 12.9716, lon: 77.5946 },
  { id: "hyderabad", label: "Hyderabad", lat: 17.3850, lon: 78.4867 },
  { id: "ahmedabad", label: "Ahmedabad", lat: 23.0225, lon: 72.5714 },
  { id: "kochi", label: "Kochi", lat: 9.9312, lon: 76.2673 },
];

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
  const center = metadata?.georeference?.center ?? {
    lat: 22,
    lon: 83,
  };

  // Fetch weather for all Indian cities but tolerate per-city failures
  const cities: any[] = [];
  for (const city of INDIAN_CITIES) {
    try {
      const data = await fetchOpenMeteo(city.lat, city.lon);
      cities.push({ id: city.id, label: city.label, lat: city.lat, lon: city.lon, ...data });
    } catch (err) {
      // Don't fail entire request for one failing city. Return nulls for that city.
      cities.push({ id: city.id, label: city.label, lat: city.lat, lon: city.lon, temperature_c: null, humidity_pct: null, wind_kph: null, rain_mm: null, rain_probability_pct: null });
    }
  }

  return NextResponse.json({ source: "open-meteo", center, cities }, { headers: { "Cache-Control": "no-store" } });
}