// https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/marian.rs

#![allow(unused)]
use candle_core::{Module, Result, Tensor};
use candle_transformers::quantized_var_builder;
use candle_nn::{layer_norm, LayerNorm, VarBuilder};
use candle_nn;
use std::sync::Arc;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub decoder_vocab_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_attention_heads: usize,
    pub use_cache: bool,
    pub is_encoder_decoder: bool,
    pub activation_function: candle_nn::Activation,
    pub d_model: usize,
    pub decoder_start_token_id: u32,
    pub scale_embedding: bool,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub forced_eos_token_id: u32,
    pub share_encoder_decoder_embeddings: bool,
}

impl Config {
    // https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-fr-en/blob/main/config.json
    pub fn opus_mt_tc_big_fr_en() -> Self {
        Self {
            activation_function: candle_nn::Activation::Relu,
            d_model: 1024,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            decoder_layers: 6,
            decoder_start_token_id: 53016,
            decoder_vocab_size: Some(53017),
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            encoder_layers: 6,
            eos_token_id: 43311,
            forced_eos_token_id: 43311,
            is_encoder_decoder: true,
            max_position_embeddings: 1024,
            pad_token_id: 53016,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            use_cache: true,
            vocab_size: 53017,
        }
    }

    // https://huggingface.co/Helsinki-NLP/opus-mt-fr-en/blob/main/config.json
    pub fn opus_mt_fr_en() -> Self {
        Self {
            activation_function: candle_nn::Activation::Relu, // TODO should be Swish
            d_model: 512,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 2048,
            decoder_layers: 6,
            decoder_start_token_id: 59513,
            decoder_vocab_size: Some(59514),
            encoder_attention_heads: 8,
            encoder_ffn_dim: 2048,
            encoder_layers: 6,
            eos_token_id: 0,
            forced_eos_token_id: 0,
            is_encoder_decoder: true,
            max_position_embeddings: 512,
            pad_token_id: 59513,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            use_cache: true,
            vocab_size: 59514,
        }
    }
}

#[derive(Debug, Clone)]
struct SinusoidalPositionalEmbedding {
    emb: Embedding,
}

impl SinusoidalPositionalEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device();
        let dtype = vb.dtype();
        let num_positions = cfg.max_position_embeddings;
        let dim = cfg.d_model;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, num_positions as u32, dev)?
            .to_dtype(dtype)?
            .reshape((num_positions, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let weights = Tensor::cat(&[&sin, &cos], 1)?.contiguous()?;
        let emb = Embedding::from_weights(weights)?;
        Ok(Self { emb })
    }

    fn forward(&self, input_ids: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        Tensor::arange(
            past_kv_len as u32,
            (past_kv_len + seq_len) as u32,
            input_ids.device(),
        )?
        .apply(&self.emb)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scaling: f64,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, is_decoder: bool, vb: VarBuilder) -> Result<Self> {
        let num_heads = if is_decoder {
            cfg.decoder_attention_heads
        } else {
            cfg.encoder_attention_heads
        };
        let embed_dim = cfg.d_model;
        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scaling,
            num_heads,
            head_dim,
        })
    }

    fn _shape(&self, tensor: &Tensor, bsz: usize) -> Result<Tensor> {
        tensor
            .reshape((bsz, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let is_cross_attn = kv_states.is_some();
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?;
        let (key_states, value_states) = match kv_states {
            None => {
                let key_states = self._shape(&xs.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&xs.apply(&self.v_proj)?, b_sz)?;
                (key_states, value_states)
            }
            Some(kv_states) => {
                let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                (key_states, value_states)
            }
        };
        let proj_shape = (b_sz * self.num_heads, (), self.head_dim);
        let query_states = self._shape(&query_states, b_sz)?.reshape(proj_shape)?;
        let key_states = key_states.reshape(proj_shape)?;
        let value_states = value_states.reshape(proj_shape)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(attn_mask) => attn_weights.broadcast_add(attn_mask)?,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value_states)?;
        attn_output
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.head_dim * self.num_heads))?
            .apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, true, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn: cfg.activation_function,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = (self.self_attn.forward(xs, None, None)? + residual)?
            .apply(&self.self_attn_layer_norm)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)?;
        (xs + residual)?.apply(&self.final_layer_norm)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, true, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = Attention::new(cfg, true, vb.pp("encoder_attn"))?;
        let encoder_attn_layer_norm =
            layer_norm(cfg.d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear(cfg.d_model, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.decoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn: cfg.activation_function,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = (self.self_attn.forward(xs, None, Some(attn_mask))? + residual)?
            .apply(&self.self_attn_layer_norm)?;
        let xs = match encoder_xs {
            None => xs,
            Some(encoder_xs) => {
                let residual = &xs;
                let xs = self.encoder_attn.forward(&xs, Some(encoder_xs), None)?;
                (residual + xs)?.apply(&self.encoder_attn_layer_norm)?
            }
        };
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)?;
        let xs = (xs + residual)?.apply(&self.final_layer_norm)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<EncoderLayer>,
    embed_scale: Option<f64>,
}

impl Encoder {
    fn new(cfg: &Config, embed_tokens: &Embedding, vb: VarBuilder) -> Result<Self> {
        let embed_positions = SinusoidalPositionalEmbedding::new(cfg, vb.pp("embed_positions"))?;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.encoder_layers {
            let layer = EncoderLayer::new(cfg, vb_l.pp(idx))?;
            layers.push(layer)
        }
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
            embed_scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let xs = xs.apply(&self.embed_tokens)?;
        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };
        let embed_pos = self
            .embed_positions
            .forward(&xs, past_kv_len)?
            .unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&embed_pos)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<DecoderLayer>,
    embed_scale: Option<f64>,
}

impl Decoder {
    fn new(cfg: &Config, embed_tokens: &Embedding, vb: VarBuilder) -> Result<Self> {
        let embed_positions = SinusoidalPositionalEmbedding::new(cfg, vb.pp("embed_positions"))?;
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.decoder_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(idx))?;
            layers.push(layer)
        }
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
            embed_scale,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = xs.apply(&self.embed_tokens)?;
        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };
        let embed_pos = self
            .embed_positions
            .forward(&xs, past_kv_len)?
            .unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&embed_pos)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, encoder_xs, attn_mask)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Model {
    shared: Embedding,
    encoder: Encoder,
    decoder: Decoder,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, vb.pp("shared"))?;
        let encoder = Encoder::new(cfg, &shared, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, &shared, vb.pp("decoder"))?;
        Ok(Self {
            shared,
            encoder,
            decoder,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MTModel {
    model: Model,
    lm_head: Linear,
    final_logits_bias: Tensor,
}

impl MTModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let target_vocab_size = cfg.decoder_vocab_size.unwrap_or(cfg.vocab_size);
        let final_logits_bias = vb.get((1, target_vocab_size), "final_logits_bias")?;
        let model = Model::new(cfg, vb.pp("model"))?;
        let lm_head = Linear::from_weights(model.shared.embeddings().clone(), None);
        Ok(Self {
            model,
            lm_head,
            final_logits_bias,
        })
    }

    pub fn encoder(&self) -> &Encoder {
        &self.model.encoder
    }

    pub fn decoder(&self) -> &Decoder {
        &self.model.decoder
    }

    pub fn decode(&self, xs: &Tensor, encoder_xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), xs.device())?;
        self.model
            .decoder
            .forward(xs, Some(encoder_xs), 0, &mask)?
            .apply(&self.lm_head)?
            .broadcast_add(&self.final_logits_bias)
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    span: tracing::Span,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}


