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

  const backendUrl = apiOrigin ? `${apiOrigin}/forecast/latest/${filename}` : "";
  const gcsUrl = `https://storage.googleapis.com/${bucket}/${prefix}/${filename}`;
  const candidateUrls = backendUrl ? [backendUrl, gcsUrl] : [gcsUrl];

  let lastStatus: number | null = null;
  let lastError: string | null = null;

  for (const targetUrl of candidateUrls) {
    try {
      const resp = await fetch(targetUrl, { cache: "no-store" });
      if (!resp.ok) {
        lastStatus = resp.status;
        lastError = `Forecast image fetch failed with ${resp.status}`;
        continue;
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
      lastError = error instanceof Error ? error.message : "Unknown image proxy error";
    }
  }

  return NextResponse.json(
    {
      error: lastError || "Unable to fetch forecast image",
      status: lastStatus,
      attempted: candidateUrls,
    },
    { status: 502 },
  );
}
