import { NextRequest, NextResponse } from "next/server";

const PY_BASE = process.env.PY_BACKEND_URL ?? "http://127.0.0.1:8000";

async function forward(req: NextRequest) {
  const url = new URL(req.url);
  const target = `${PY_BASE}${url.pathname}${url.search}`;

  const body = await req.arrayBuffer();

  const res = await fetch(target, {
    method: req.method,
    headers: {
      "content-type": req.headers.get("content-type") ?? "",
    },
    body: body.byteLength ? body : undefined,
  });

  const contentType = res.headers.get("content-type") ?? "";
  const data = contentType.includes("application/json")
    ? await res.json()
    : await res.text();

  return NextResponse.json(data, { status: res.status });
}

export async function POST(req: NextRequest) { return forward(req); }
export async function GET(req: NextRequest) { return forward(req); }
export async function DELETE(req: NextRequest) { return forward(req); }
