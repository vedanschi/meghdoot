import { NextResponse } from "next/server";

function trimSlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, "");
}

export async function GET() {
  const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
  const prefix = trimSlashes(process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest");
  const apiOrigin = (process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_ORIGIN || "").replace(/\/$/, "");

  const targetUrl = apiOrigin
    ? `${apiOrigin}/forecast/latest/metadata.json`
    : `https://storage.googleapis.com/${bucket}/${prefix}/metadata.json`;

  try {
    const resp = await fetch(targetUrl, { cache: "no-store" });
    if (!resp.ok) {
      return NextResponse.json(
        { error: `Metadata fetch failed with ${resp.status}`, targetUrl },
        { status: 502 },
      );
    }

    const raw = await resp.text();
    return new NextResponse(raw, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown metadata proxy error";
    return NextResponse.json({ error: message, targetUrl }, { status: 500 });
  }
}
