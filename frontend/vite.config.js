import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const backend = "http://127.0.0.1:8000"
const apiProxy = {
  "/chat": { target: backend, changeOrigin: true },
  "/health": { target: backend, changeOrigin: true },
}

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: apiProxy,
  },
  preview: {
    proxy: apiProxy,
  },
})
