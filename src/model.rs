use {
    std::time::Instant,
    tracing::info,
    candle_core::{Device, DType, Tensor},
    candle_nn::VarBuilder,
    candle_transformers::{models::llama::{self, Llama, LlamaConfig}, generation::LogitsProcessor},
    tokenizers::Tokenizer,
    rand::Rng,
};

// based on https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs
pub struct ChatModel {
    llama: Llama,
    tokenizer: Tokenizer,
    device: Device,

    eos_token_id: u32,
}

pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(PartialEq, Eq)]
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
            "meta-llama/Llama-2-7b-chat-hf".to_owned(),
            RepoType::Model,
            "main".to_owned()
        ));

        let tokenizer_filename = hf_api.get("tokenizer.json").unwrap();
        let config_filename = hf_api.get("config.json").unwrap();
        let filenames = hub_load_safetensors(&hf_api, "model.safetensors.index.json");

        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename).unwrap()).unwrap();
        let config = config.into_config(false);

        let cache = llama::Cache::new(false, dtype, &config, &device).unwrap();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };

        let llama = Llama::load(vb, &cache, &config).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        Self {
            eos_token_id: tokenizer.token_to_id("</s>").unwrap(),

            llama,
            tokenizer,
            device,
        }
    }

    pub fn chat_completions(&self, chat: Vec<ChatMessage>) -> ChatMessage {
        let mut tokens = Vec::new();

        for message in chat.chunks(2) {
            if message[0].role != ChatRole::User {
                panic!("expected role of message to be user");
            }
            if message.len() > 1 && message[1].role != ChatRole::Assistant {
                panic!("expected role of message to be assistant");
            }

            let prompt = if message.len() == 1 {
                format!("[INST] {} [/INST] ", message[0].content)
            } else {
                format!("[INST] {} [/INST] {} ", message[0].content, message[1].content)
            };

            let mut message_tokens = self.tokenizer
                .encode(prompt, true)
                .unwrap()
                .get_ids()
                .to_vec();

            tokens.append(&mut message_tokens);
        }

        let mut output_tokens = Vec::new();

        let use_kv_cache = false;

        let max_tokens = 10000;
        let mut index_pos = 0;

        let mut logits_processor = LogitsProcessor::new(rand::thread_rng().gen(), Some(0.7), None);

        let started_at = Instant::now();
        for index in 0..max_tokens {
            let (context_size, context_index) = if use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).unwrap();
            let logits = self.llama.forward(&input, context_index).unwrap();
            let logits = logits.squeeze(0).unwrap();
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);
            output_tokens.push(next_token);

            if next_token == self.eos_token_id {
                break;
            }
        }
        info!("finished generating response: {} tokens, {}s", tokens.len(), (Instant::now() - started_at).as_secs());

        ChatMessage {
            role: ChatRole::Assistant,
            content: self.tokenizer.decode(&output_tokens, true).unwrap(),
        }
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
