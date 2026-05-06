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

export async function GET() {
  const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
  const prefix = trimSlashes(process.env.NEXT_PUBLIC_FORECAST_PREFIX || "forecasts/latest");
  const apiOrigin = (process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_ORIGIN || "").replace(/\/$/, "");

  const backendUrl = apiOrigin ? `${apiOrigin}/forecast/latest/metadata.json` : "";
  const gcsUrl = `https://storage.googleapis.com/${bucket}/${prefix}/metadata.json`;
  const candidateUrls = backendUrl ? [backendUrl, gcsUrl] : [gcsUrl];

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
        lastError = `Metadata fetch failed with ${resp.status}`;
        continue;
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
      lastError = error instanceof Error ? error.message : "Unknown metadata proxy error";
    }
  }

  return NextResponse.json(
    {
      error: lastError || "Unable to fetch metadata",
      status: lastStatus,
      attempted: candidateUrls,
    },
    { status: 502 },
  );
}
