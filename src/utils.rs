// based on https://github.com/metrics-rs/metrics/blob/main/metrics-util/src/layers/prefix.rs

use {
    metrics::{Recorder, KeyName, Unit, SharedString, Key, Metadata},
    metrics_util::layers::Layer,
};

const PREFIX: &str = "llama_inference_";

pub struct Prefix<R> {
    inner: R,
}

impl<R> Prefix<R> {
    fn prefix_key(&self, key: &Key) -> Key {
        let mut new_name = String::with_capacity(PREFIX.len() + key.name().len());
        new_name.push_str(PREFIX);
        new_name.push_str(key.name());

        Key::from_parts(new_name, key.labels())
    }

    fn prefix_key_name(&self, key_name: KeyName) -> KeyName {
        let mut new_name = String::with_capacity(PREFIX.len() + key_name.as_str().len());
        new_name.push_str(PREFIX);
        new_name.push_str(key_name.as_str());

        KeyName::from(new_name)
    }
}

impl<R: Recorder> Recorder for Prefix<R> {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_counter(self.prefix_key_name(key), unit, description)
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_gauge(self.prefix_key_name(key), unit, description)
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_histogram(self.prefix_key_name(key), unit, description)
    }

    fn register_counter(&self, key: &Key, metadata: &Metadata<'_>) -> metrics::Counter {
        self.inner.register_counter(&self.prefix_key(key), metadata)
    }

    fn register_gauge(&self, key: &Key, metadata: &Metadata<'_>) -> metrics::Gauge {
        self.inner.register_gauge(&self.prefix_key(key), metadata)
    }

    fn register_histogram(&self, key: &Key, metadata: &Metadata<'_>) -> metrics::Histogram {
        self.inner.register_histogram(&self.prefix_key(key), metadata)
    }
}

pub struct PrefixLayer;

impl<R> Layer<R> for PrefixLayer {
    type Output = Prefix<R>;

    fn layer(&self, inner: R) -> Self::Output {
        Prefix { inner }
    }
}
