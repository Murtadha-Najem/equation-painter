# Equation Painter

Upload a picture and get it back redrawn from soft ellipses, with the full formula that paints every pixel written underneath, in the style of the mathematical artist Hamid Naderi Yeganeh.

Everything runs in the browser. The picture never leaves the device.

## How it works

Each pixel (m, n) gets the colour rgb(F(H0), F(H1), F(H2)). H_v is a linear gradient with N soft ellipses laid over it. Each ellipse is a mask e^(-e^z), close to 1 inside and 0 outside, and its colour is a quadratic in v that passes through the red, green and blue values at v = 0, 1, 2.

A greedy search adds one ellipse at a time: random tries, biased toward the pixels furthest from the picture, then small nudges to the best one. Constants are rounded to two decimals during the search, and the final picture is computed again from the printed numbers, so the formula shown is exactly the one that drew it.

The formula stores no pixels. Each term is eight numbers. The search does read the picture to measure the difference, as any automatic fit must.
