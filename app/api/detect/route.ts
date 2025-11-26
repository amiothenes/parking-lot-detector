import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { image } = await request.json();

    if (!image) {
      return NextResponse.json(
        { error: 'Image is required' },
        { status: 400 }
      );
    }

    const apiKey = process.env.ROBOFLOW_API_KEY;
    
    if (!apiKey) {
      return NextResponse.json(
        { error: 'API key not configured' },
        { status: 500 }
      );
    }

    const response = await fetch(
      'https://serverless.roboflow.com/victor-c99mj/workflows/find-cars-black-jetta-cars-and-blue-cars-3',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key: apiKey,
          inputs: {
            image: { type: 'url', value: image },
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Roboflow API error: ${response.statusText}`);
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Detection error:', error);
    return NextResponse.json(
      { error: 'Failed to process detection' },
      { status: 500 }
    );
  }
}
