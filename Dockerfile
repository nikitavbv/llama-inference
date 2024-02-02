FROM frolvlad/alpine-glibc:glibc-2.34
WORKDIR /app

RUN apk update && apk add libstdc++

COPY target/release/llama-inference /app/app

ENTRYPOINT ["/app/app"]
