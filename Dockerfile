FROM frolvlad/alpine-glibc:glibc-2.34
WORKDIR /app

RUN apk add libstdc++6

COPY target/release/llama-inference /app/app

ENTRYPOINT ["/app/app"]
