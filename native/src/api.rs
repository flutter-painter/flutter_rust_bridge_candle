#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
//use clap::{Parser, ValueEnum};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
mod marian;
use tokenizers::Tokenizer;
//use std::io::Write;
//use candle_transformers::generation::LogitsProcessor;
const DTYPE: DType = DType::F32;

#[derive(Clone, Debug, Copy)]
enum Which {
    Base,
    Big,
}

// #[derive(Parser)]
// struct Args {
//     #[arg(long)]
//     model: Option<String>,

//     #[arg(long)]
//     tokenizer: Option<String>,

//     #[arg(long)]
//     tokenizer_dec: Option<String>,

//     /// Choose the variant of the model to run.
//     #[arg(long, default_value = "big")]
//     which: Which,

//     /// Run on CPU rather than on GPU.
//     #[arg(long)]
//     cpu: bool,

//     /// Use the quantized version of the model.
//     #[arg(long)]
//     quantized: bool,

//     /// Text to be translated
//     #[arg(long)]
//     text: String,
// }

fn device(cpu: bool) -> Result<Device, E> {
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

pub fn translate(text: String) -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;

    let config = marian::Config::opus_mt_fr_en();
    // let config = match args.which {
    //     Which::Base => marian::Config::opus_mt_fr_en(),
    //     Which::Big => marian::Config::opus_mt_tc_big_fr_en(),
    // };
    let tokenizer = {
        let tokenizer = {
                let name =  "tokenizer-marian-base-fr.json";
                // let name = match args.which {
                //     Which::Base => "tokenizer-marian-base-fr.json",
                //     Which::Big => "tokenizer-marian-fr.json",
                // };
                Api::new()?
                    .model("lmz/candle-marian".to_string())
                    .get(name)?

        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };

    let tokenizer_dec = {
        let tokenizer = {
                let name =  "tokenizer-marian-base-en.json"
                ;
                // let name = match args.which {
                //     Which::Base => "tokenizer-marian-base-en.json",
                //     Which::Big => "tokenizer-marian-en.json",
                // };
                Api::new()?
                    .model("lmz/candle-marian".to_string())
                    .get(name)?
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };

    let device = device(true)?;
    //let model_cache = std::path::PathBuf::from(model);
    let vb = {
        let model =  Api::new()?
                    .repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-fr-en".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/4".to_string(),
                    ))
                    .get("model.safetensors")?
        ;
        // let model = match args.model {
        //     Some(model) => std::path::PathBuf::from(model),
        //     None => match args.which {
        //         Which::Base => Api::new()?
        //             .repo(hf_hub::Repo::with_revision(
        //                 "Helsinki-NLP/opus-mt-fr-en".to_string(),
        //                 hf_hub::RepoType::Model,
        //                 "refs/pr/4".to_string(),
        //             ))
        //             .get("model.safetensors")?,
        //         Which::Big => Api::new()?
        //             .model("Helsinki-NLP/opus-mt-tc-big-fr-en".to_string())
        //             .get("model.safetensors")?,
        //     },
        // };
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model], DType::F32, &device)? }
    };
    let model = marian::MTModel::new(&config, vb)?;

    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let encoder_xs = {
        let mut tokens = tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.push(config.eos_token_id);
        let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        model.encoder().forward(&tokens, 0)?
    };

    let mut token_ids = vec![config.decoder_start_token_id];
    for index in 0..1000 {
        // TODO: Add a kv cache.
        let context_size = if index >= 1000 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.decode(&input_ids, &encoder_xs)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);
        println!("{token}");
        if token == config.eos_token_id || token == config.forced_eos_token_id {
            break;
        }
    }
    println!(
        "{}",
        tokenizer_dec.decode(&token_ids, true).map_err(E::msg)?
    );
    Ok(())
}