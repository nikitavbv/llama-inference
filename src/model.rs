use {
    tracing::info,
    candle_core::{Device, DType},
    candle_nn::VarBuilder,
    candle_transformers::models::llama::{self, Llama, LlamaConfig},
    tokenizers::Tokenizer,
};

// based on https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs
pub struct ChatModel {
    llama: Llama,
    tokenizer: Tokenizer,
}

pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatModel {
    pub fn load() -> Self {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        info!("loading model");

        let device = Device::Cpu;
        let dtype = DType::F32;

        let hf_api = Api::new().unwrap();
        let hf_api = hf_api.repo(Repo::with_revision(
            "meta-llama/Llama-2-7b-hf".to_owned(),
            RepoType::Model,
            "main".to_owned()
        ));

        let tokenizer_filename = hf_api.get("tokenizer.json").unwrap();
        let config_filename = hf_api.get("config.json").unwrap();
        let filenames = hub_load_safetensors(&hf_api, "model.safetensors.index.json");

        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename).unwrap()).unwrap();
        let config = config.into_config(false);

        let cache = llama::Cache::new(true, dtype, &config, &device).unwrap();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };

        let llama = Llama::load(vb, &cache, &config).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        Self {
            llama,
            tokenizer,
        }
    }

    pub fn chat_completions(&self, chat: Vec<ChatMessage>) -> ChatMessage {
        unimplemented!()
    }
}

// taken from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs#L122
/// Loads the safetensors files for a model from the hub based on a json index file.
fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Vec<std::path::PathBuf> {
    let json_file = repo.get(json_file).unwrap();
    let json_file = std::fs::File::open(json_file).unwrap();
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).unwrap();
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    safetensors_files
        .iter()
        .map(|v| repo.get(v).unwrap())
        .collect::<Vec<_>>()
}
