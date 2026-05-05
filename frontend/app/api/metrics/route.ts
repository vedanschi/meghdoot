import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

type MetricsPayload = {
  diffusion: Record<string, number>;
  convlstm: Record<string, number>;
};

export async function GET() {
  try {
    const metricsPath = path.resolve(
      process.cwd(),
      "..",
      "src",
      "meghdoot",
      "evaluation",
      "metrics.json",
    );
    const raw = await fs.readFile(metricsPath, "utf-8");
    const parsed = JSON.parse(raw) as MetricsPayload;

    return NextResponse.json(parsed, {
      headers: {
        "Cache-Control": "public, max-age=60",
      },
    });
  } catch (error) {
    // If local metrics file isn't available in the deployed frontend image,
    // fall back to fetching from the GCS bucket so the frontend can still show metrics.
    try {
      const bucket = process.env.NEXT_PUBLIC_GCS_BUCKET || "meghdoot-satellite-data";
      const url = `https://storage.googleapis.com/${bucket}/metrics.json`;
      const resp = await fetch(url, { cache: "no-store" });
      if (!resp.ok) {
        const msg = `Metrics fetch failed with ${resp.status}`;
        return NextResponse.json({ error: msg }, { status: 502 });
      }
      const parsed = await resp.json();
      return NextResponse.json(parsed, {
        headers: { "Cache-Control": "public, max-age=60" },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown metrics read error";
      return NextResponse.json({ error: message }, { status: 500 });
    }
  }
}
