# GreenNet Frontend

This directory is the official entrypoint for the GreenNet UI.

Run commands from here:

```bash
npm run dev
npm run build
npm run lint
```

Implementation notes:
- The React/Vite source lives in `greennet-ui/`.
- The supported product/demo routes are `/dashboard`, `/results`, and `/simulator`.
- The Streamlit app under `/dashboard` is internal tooling, not the official frontend surface.
