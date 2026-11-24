# ----------------------------- build stage -----------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first so this layer is cached between source edits.
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .

# Vite inlines the API URL at build time, so it has to be a build argument
# rather than a runtime environment variable.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---------------------------- runtime stage ----------------------------
FROM nginx:alpine AS runtime

# Serve the static bundle with SPA fallback on port 3000.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
