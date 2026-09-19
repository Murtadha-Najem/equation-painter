# Equation Painter

Upload a picture and get it back redrawn from soft ellipses, with the full formula that paints every pixel written underneath, in the style of the mathematical artist Hamid Naderi Yeganeh.

Everything runs in the browser. The picture never leaves the device.

## How it works

Each pixel (m, n) gets the colour rgb(F(H0), F(H1), F(H2)). H_v is a linear gradient with N soft ellipses laid over it. Each ellipse is a mask e^(-e^z), close to 1 inside and 0 outside, and its colour is a quadratic in v that passes through the red, green and blue values at v = 0, 1, 2.

Each term is a soft, bendable ellipse with its own edge sharpness, and its colour is alpha + gamma U + delta W per channel (U, W its own axes), so every term carries its own shading. A greedy search adds one term at a time: 64 random candidates at the places furthest from the picture, scored at half resolution with colours solved exactly by least squares, then nudged at full resolution inside the term's own window. Terms are never smaller than a pixel and a half of the picture it reads, and there is at most one term for every 40 of its pixels; when the budget allows more, each term is searched harder instead. Constants are rounded to three decimals during the search, and the final picture is computed again from the printed numbers, so the formula shown is exactly the one that drew it.

The formula stores no pixels. Each term is sixteen numbers. The search does read the picture to measure the difference, as any automatic fit must.

## Paste an equation

`paste.html` takes a formula as plain text, draws it, and redraws as you edit. It reads four kinds:

- `mode: pixel`: every pixel from `H`, with `v = 0, 1, 2` for red, green and blue (Yeganeh's Frog is included).
- `mode: circles`: a family of circles with centre `(A, B)` and radius `R` for each `k` (his A Bird in Flight).
- `mode: segments`: a family of line segments from `(X1, Y1)` to `(X2, Y2)` for each `k` (his Boat and 1,000,000 Line Segments).
- `mode: layers`: the painter's own output, copied with "Copy formula" or opened with "Open in Paste an equation".

The painter also records a 10 second video of the picture forming, as MP4 where the browser supports it.
