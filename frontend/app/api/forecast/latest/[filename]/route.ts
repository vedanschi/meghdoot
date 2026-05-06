import { NextResponse } from "next/server";

function trimSlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, "");
}

export async function GET(
  _: Request,
  context: { params: { filename: string } },
) {
  const { filename } = context.params;
  if (!filename.startsWith("forecast_step_") || !filename.endsWith(".png")) {
    return NextResponse.json({ error: "Unknown forecast artifact" }, { status: 404 });
  }

  const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
  const prefix = trimSlashes(process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest");
  const apiOrigin = (process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_ORIGIN || "").replace(/\/$/, "");

  const targetUrl = apiOrigin
    ? `${apiOrigin}/forecast/latest/${filename}`
    : `https://storage.googleapis.com/${bucket}/${prefix}/${filename}`;

  try {
    const resp = await fetch(targetUrl, { cache: "no-store" });
    if (!resp.ok) {
      return NextResponse.json(
        { error: `Forecast image fetch failed with ${resp.status}`, targetUrl },
        { status: 502 },
      );
    }

    const bytes = await resp.arrayBuffer();
    return new NextResponse(bytes, {
      status: 200,
      headers: {
        "Content-Type": resp.headers.get("content-type") || "image/png",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown image proxy error";
    return NextResponse.json({ error: message, targetUrl }, { status: 500 });
  }
}
