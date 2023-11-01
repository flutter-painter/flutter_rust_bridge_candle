// This is the entry point of your Rust library.
// When adding new code to your project, note that only items used
// here will be transformed to their Dart equivalents.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::path::PathBuf;

use candle_transformers::models::t5;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// use std::io::Write;
//use candle_transformers::generation::LogitsProcessor;
//const DTYPE: DType = DType::F32;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Enable decoding.
    #[arg(long)]
    decode: bool,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: Option<String>,

    /// If set along with --decode, will use this prompt to initialize the decoder.
    #[arg(long)]
    decoder_prompt: Option<String>,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}
// taken from hugg face example
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}
pub struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}



    pub fn load() -> Result<(T5ModelBuilder, Tokenizer)> {
        let device = device(true)?;
        let default_model = "t5-small".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = (default_model, default_revision);
        

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = if model_id == "google/flan-t5-xxl" {
            vec![
                api.get("model-00001-of-00005.safetensors")?,
                api.get("model-00002-of-00005.safetensors")?,
                api.get("model-00003-of-00005.safetensors")?,
                api.get("model-00004-of-00005.safetensors")?,
                api.get("model-00005-of-00005.safetensors")?,
            ]
        } else {
            vec![api.get("model.safetensors")?]
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !false;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            T5ModelBuilder {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(t5: &T5ModelBuilder) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&t5.weights_filename, DTYPE, &t5.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &t5.config)?)
    }

    pub fn build_conditional_generation(t5: &T5ModelBuilder) -> Result<t5::T5ForConditionalGeneration> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&t5.weights_filename, DTYPE, &t5.device)?
        };
        Ok(t5::T5ForConditionalGeneration::load(vb, &t5.config)?)
    }


pub fn translate(prompt: Option<String>) -> Result<()> {
    ////use tracing_chrome::ChromeLayerBuilder;
    ////use tracing_subscriber::prelude::*;
    // let args = Args::parse();

    // let _guard = if args.tracing {
    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    //     tracing_subscriber::registry().with(chrome_layer).init();
    //     Some(guard)
    // } else {
    //     None
    // };

    let (builder, mut tokenizer) = load()?;
    let device = &builder.device;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    match prompt {
        Some(prompt) => {
            let tokens = tokenizer
                .encode(prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

                let mut model = build_encoder(&builder)?;
                let start = std::time::Instant::now();
                let ys = model.forward(&input_token_ids)?;
                println!("{ys}");
                println!("Took {:?}", start.elapsed());
            
        }
        None => println!("none")
    }
    Ok(())
}




// A plain enum without any fields. This is similar to Dart- or C-style enums.
// flutter_rust_bridge is capable of generating code for enums with fields
// (@freezed classes in Dart and tagged unions in C).
pub enum Platform {
    Unknown,
    Android,
    Ios,
    Windows,
    Unix,
    MacIntel,
    MacApple,
    Wasm,
}

// A function definition in Rust. Similar to Dart, the return type must always be named
// and is never inferred.
pub fn platform() -> Platform {
    // This is a macro, a special expression that expands into code. In Rust, all macros
    // end with an exclamation mark and can be invoked with all kinds of brackets (parentheses,
    // brackets and curly braces). However, certain conventions exist, for example the
    // vector macro is almost always invoked as vec![..].
    //
    // The cfg!() macro returns a boolean value based on the current compiler configuration.
    // When attached to expressions (#[cfg(..)] form), they show or hide the expression at compile time.
    // Here, however, they evaluate to runtime values, which may or may not be optimized out
    // by the compiler. A variety of configurations are demonstrated here which cover most of
    // the modern oeprating systems. Try running the Flutter application on different machines
    // and see if it matches your expected OS.
    //
    // Furthermore, in Rust, the last expression in a function is the return value and does
    // not have the trailing semicolon. This entire if-else chain forms a single expression.
    if cfg!(windows) {
        Platform::Windows
    } else if cfg!(target_os = "android") {
        Platform::Android
    } else if cfg!(target_os = "ios") {
        Platform::Ios
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Platform::MacApple
    } else if cfg!(target_os = "macos") {
        Platform::MacIntel
    } else if cfg!(target_family = "wasm") {
        Platform::Wasm
    } else if cfg!(unix) {
        Platform::Unix
    } else {
        Platform::Unknown
    }
}

// The convention for Rust identifiers is the snake_case,
// and they are automatically converted to camelCase on the Dart side.
pub fn rust_release_mode() -> bool {
    cfg!(not(debug_assertions))
}

