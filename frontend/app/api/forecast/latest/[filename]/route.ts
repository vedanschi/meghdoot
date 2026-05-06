import { NextResponse } from "next/server";

function trimSlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, "");
}

async function getIdentityToken(audience: string): Promise<string | null> {
  try {
    const url = new URL(
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
    );
    url.searchParams.set("audience", audience);
    url.searchParams.set("format", "full");

    const response = await fetch(url, {
      headers: { "Metadata-Flavor": "Google" },
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }

    return await response.text();
  } catch {
    return null;
  }
}

export async function GET(
  _: Request,
  context: { params: { filename: string } },
) {
  const { filename } = context.params;
  const allowedName = filename === "current.png" || (filename.startsWith("forecast_step_") && filename.endsWith(".png"));
  if (!allowedName) {
    return NextResponse.json({ error: "Unknown forecast artifact" }, { status: 404 });
  }

  const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
  const prefix = trimSlashes(process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest");
  const apiOrigin = (process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_ORIGIN || "").replace(/\/$/, "");

  const backendUrl = apiOrigin ? `${apiOrigin}/forecast/latest/${filename}` : "";
  const gcsUrl = `https://storage.googleapis.com/${bucket}/${prefix}/${filename}`;
  let candidateUrls = backendUrl ? [backendUrl, gcsUrl] : [gcsUrl];
  
  // For current.png, add fallback to forecast_step_0.png
  if (filename === "current.png") {
    const backendFallback = apiOrigin ? `${apiOrigin}/forecast/latest/forecast_step_0.png` : "";
    const gcsFallback = `https://storage.googleapis.com/${bucket}/${prefix}/forecast_step_0.png`;
    if (backendFallback) {
      candidateUrls.push(backendFallback);
    }
    candidateUrls.push(gcsFallback);
  }

  let lastStatus: number | null = null;
  let lastError: string | null = null;

  for (const targetUrl of candidateUrls) {
    try {
      const headers: Record<string, string> = {};
      if (targetUrl === backendUrl && apiOrigin) {
        const token = await getIdentityToken(apiOrigin);
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }
      }

      const resp = await fetch(targetUrl, { cache: "no-store", headers });
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
