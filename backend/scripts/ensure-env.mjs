import { copyFileSync, existsSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const root = join(dirname(fileURLToPath(import.meta.url)), "..")
const envPath = join(root, ".env")
const examplePath = join(root, ".env.example")

if (!existsSync(envPath)) {
  copyFileSync(examplePath, envPath)
  console.log("Created backend/.env from .env.example — edit .env and set GEMINI_API_KEY.")
}
