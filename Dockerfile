FROM golang:1.26.2-alpine AS builder

WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o femtollm ./cmd/femtollm

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/femtollm /usr/local/bin/femtollm

# Seed config (always applied on startup so GitHub is the source of truth)
COPY config.example.json /app/config.default.json
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["femtollm"]
