FROM ubuntu:22.04
WORKDIR /app

COPY target/release/llama-inference /app/app

ENTRYPOINT ["/app/app"]
