FROM golang:1.24-alpine AS builder

WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o femtollm ./cmd/femtollm

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/femtollm /usr/local/bin/femtollm

EXPOSE 8080
CMD ["femtollm"]
