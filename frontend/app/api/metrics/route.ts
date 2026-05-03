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
    const message = error instanceof Error ? error.message : "Unknown metrics read error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
