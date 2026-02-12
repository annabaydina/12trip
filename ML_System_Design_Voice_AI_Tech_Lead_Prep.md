# ML System Design for Voice AI Engineer Tech Lead -- Complete Preparation Guide

## Table of Contents

1. [The Interview Framework (Use This Every Time)](#1-the-interview-framework)
2. [Voice AI Domain Knowledge You MUST Know](#2-voice-ai-domain-knowledge)
3. [Core Voice AI System Designs (Top 10 Questions)](#3-core-voice-ai-system-designs)
4. [Architecture Patterns & Building Blocks](#4-architecture-patterns)
5. [Key Metrics & Evaluation](#5-key-metrics)
6. [Scaling & Production Concerns](#6-scaling--production)
7. [Tech Lead Specific: Leadership Signals](#7-tech-lead-signals)
8. [Speed-Run Preparation Plan (2-Week Blitz)](#8-speed-run-preparation)
9. [Cheat Sheets & Quick References](#9-cheat-sheets)
10. [Mock Interview Scripts](#10-mock-interview-scripts)

---

## 1. The Interview Framework

### The 45-Minute Structure (Memorize This)

Every ML system design answer should follow this skeleton. Practice until it becomes automatic.

```
[0-5 min]   CLARIFY   -- Requirements, scope, constraints
[5-12 min]  HIGH-LEVEL DESIGN -- End-to-end architecture diagram
[12-25 min] DEEP DIVE -- Model choices, data pipeline, feature engineering
[25-35 min] SCALING & PRODUCTION -- Serving, monitoring, failure modes
[35-42 min] TRADE-OFFS & ALTERNATIVES -- Why X over Y
[42-45 min] EXTENSIONS -- Future improvements, what you'd do with more time
```

### Step 1: CLARIFY (The Most Important 5 Minutes)

**Always ask these questions:**

| Category | Questions to Ask |
|----------|-----------------|
| **Users** | Who are the end users? How many concurrent users? |
| **Scale** | How many requests per second? What latency budget? |
| **Data** | What data is available? Labeled or unlabeled? How much? |
| **Quality** | What accuracy/WER target? What's acceptable failure rate? |
| **Constraints** | On-device vs cloud? Privacy requirements? Languages? |
| **Business** | What's the primary business metric we're optimizing? |

**Voice AI Specific Clarifications:**

- Streaming vs batch processing?
- Real-time latency requirements (< 200ms for conversational AI)?
- Noise conditions (call center, mobile, car, smart speaker)?
- Speaker demographics (accents, ages, languages)?
- On-device vs cloud inference?
- Privacy constraints (HIPAA, GDPR, data residency)?

### Step 2: HIGH-LEVEL DESIGN

Draw the end-to-end pipeline. For Voice AI, the generic pipeline is:

```
Audio Input --> Preprocessing --> Feature Extraction --> Model Inference --> Post-processing --> Output
     |               |                  |                     |                  |              |
  Microphone    Noise reduction     Mel spectrograms    ASR/TTS/etc.       Beam search     Text/Audio
  Streaming     VAD (Voice         MFCC features       Transformer         Punctuation     API response
  Audio codec   Activity Detect)   Filter banks        CTC/Attention       Capitalization
```

**Trick:** Always draw the diagram FIRST, then talk through it. This gives you thinking time and shows structured thinking.

### Step 3: DEEP DIVE

This is where you show domain expertise. Pick 2-3 components to go deep on based on interviewer interest.

### Step 4: SCALING & PRODUCTION

Show you can take a model from notebook to production. Cover:
- Model serving (TorchServe, Triton, TFServing)
- Batching strategies
- Model optimization (quantization, pruning, distillation)
- Monitoring & observability
- A/B testing framework
- Rollback strategy

### Step 5: TRADE-OFFS

**Always present at least 2 alternatives and explain why you chose one.**

---

## 2. Voice AI Domain Knowledge

### The Voice AI Stack (Know All Layers)

```
Layer 7: Application        -- Voice assistants, call centers, dictation
Layer 6: Dialog Management  -- Turn-taking, context tracking, intent routing
Layer 5: NLU/NLG           -- Intent classification, entity extraction, response generation
Layer 4: ASR / TTS / STS   -- Speech-to-text, text-to-speech, speech-to-speech
Layer 3: Audio Features     -- Spectrograms, embeddings, speaker vectors
Layer 2: Signal Processing  -- Noise reduction, echo cancellation, beamforming
Layer 1: Audio I/O          -- Microphones, codecs, streaming protocols
```

### Automatic Speech Recognition (ASR) -- Deep Knowledge Required

#### Evolution (Know This Timeline)

| Era | Approach | Key Models | WER |
|-----|----------|------------|-----|
| Pre-2014 | GMM-HMM | Kaldi, HTK | 20-30% |
| 2014-2017 | Hybrid DNN-HMM | DeepSpeech, CLDNN | 10-15% |
| 2017-2020 | End-to-End | LAS, CTC, RNN-T | 5-8% |
| 2020-2023 | Large pretrained | Wav2Vec2, HuBERT, Whisper | 3-5% |
| 2023+ | Multimodal/LLM-based | USM, Gemini, GPT-4o | 2-3% |

#### Three Main E2E Architectures

**1. CTC (Connectionist Temporal Classification)**
- Pros: Simple, fast, streaming-friendly
- Cons: Conditional independence assumption, needs external LM
- Use when: Low-latency streaming required
- Key models: DeepSpeech, QuartzNet, Citrinet

**2. Attention-based Encoder-Decoder (AED / LAS)**
- Pros: Best quality, models output dependencies
- Cons: High latency (needs full utterance), harder to stream
- Use when: Offline/batch processing, quality is king
- Key models: LAS, Whisper, Speech-Transformer

**3. RNN-Transducer (RNN-T / Transducer)**
- Pros: Streaming + models output dependencies (best of both worlds)
- Cons: Complex training, slower than CTC
- Use when: Production streaming ASR (this is what Google, Apple, Amazon use)
- Key models: Google's streaming ASR, Conformer-Transducer

**Trick for interviews:** "For production streaming ASR, I'd default to a Conformer-Transducer architecture because it gives us streaming capability with joint acoustic-linguistic modeling."

#### Modern Architecture Components

**Conformer (2020) -- Know This Cold**
```
Input --> Conv Subsampling --> [Conformer Block x N] --> Output

Conformer Block:
  Feed Forward (1/2) --> Multi-Head Self-Attention --> Convolution --> Feed Forward (1/2) --> LayerNorm
```

Why Conformer wins: Combines the global context of self-attention with the local pattern recognition of convolutions. This is critical for speech where both local phonetic patterns and long-range dependencies matter.

**Self-Supervised Pre-training (Know the Big 3)**

| Model | Approach | Pre-training | Key Innovation |
|-------|----------|-------------|----------------|
| Wav2Vec 2.0 | Contrastive | Mask + predict quantized speech | Learned discrete speech units |
| HuBERT | Predictive | Mask + predict cluster assignments | Offline clustering targets |
| WavLM | Denoising | Mask + predict with noise augmentation | Robust to noise |

### Text-to-Speech (TTS) -- Know the Pipeline

#### Modern TTS Pipeline
```
Text --> Text Analysis --> Acoustic Model --> Vocoder --> Waveform
         (G2P, norm)    (Mel spectrogram)   (Audio)
```

#### Key Models

| Component | Model | Notes |
|-----------|-------|-------|
| Acoustic | Tacotron 2 | Attention-based, autoregressive |
| Acoustic | FastSpeech 2 | Non-autoregressive, parallel, faster |
| Acoustic | VITS | End-to-end, combines acoustic + vocoder |
| Acoustic | Tortoise TTS | High quality, slow |
| Vocoder | WaveNet | Autoregressive, very slow |
| Vocoder | WaveRNN | Faster autoregressive |
| Vocoder | HiFi-GAN | GAN-based, fast, high quality |
| Vocoder | WaveGrad | Diffusion-based |
| End-to-End | VALL-E | Neural codec LM, zero-shot voice cloning |
| End-to-End | Bark | GPT-style, generates audio tokens |
| End-to-End | XTTS | Multi-lingual, voice cloning |

#### Zero-Shot TTS / Voice Cloning (Hot Topic)

```
Reference Audio (3-10s) --> Speaker Encoder --> Speaker Embedding
                                                      |
Text Input --> Text Encoder ---------> Decoder -------+--> Mel Spectrogram --> Vocoder --> Audio
```

Key approaches:
- **Speaker embedding conditioned**: Extract speaker vector, condition decoder
- **In-context learning**: VALL-E style, treat speech as language modeling
- **Neural codec language models**: Encode audio into discrete tokens, model with LLM

### Voice Activity Detection (VAD)

Simple but critical component. Know these approaches:
1. **Energy-based**: Threshold on signal energy (fast, unreliable)
2. **Statistical model**: GMM on spectral features
3. **DNN-based**: Small CNN/RNN on mel features (Silero VAD, WebRTC VAD)
4. **Transformer-based**: Best quality, higher compute

### Speaker Diarization & Verification

**Speaker Verification Pipeline:**
```
Audio --> Feature Extraction --> Speaker Encoder --> Embedding --> Comparison
                                  (ECAPA-TDNN,      (d-vector,    (Cosine sim,
                                   ResNet)           x-vector)     PLDA)
```

**Speaker Diarization Pipeline:**
```
Audio --> VAD --> Segmentation --> Embedding --> Clustering --> Labels
                                  Extraction    (SC, AHC,     "Who spoke when"
                                                VBx)
```

Modern end-to-end: EEND (End-to-End Neural Diarization) handles overlap.

### Audio/Speech LLMs (Know the Frontier)

| Model | Approach | Capabilities |
|-------|----------|-------------|
| AudioPaLM | Fuse audio + text LLMs | ASR, TTS, S2S translation |
| GPT-4o | Native multimodal | Voice conversation, emotion |
| Gemini | Native multimodal | Audio understanding + generation |
| Qwen-Audio | Audio encoder + LLM | Audio understanding |
| SALMONN | Dual audio encoders + LLM | Audio + speech understanding |
| Moshi (Kyutai) | Full-duplex speech LLM | Real-time conversation |
| SpeechGPT | Discrete speech tokens + LLM | Speech-in, speech-out |

---

## 3. Core Voice AI System Designs (Top 10 Questions)

### Design 1: Real-Time Speech Recognition System (Most Common)

**Problem:** Design a real-time ASR system serving 10M daily active users.

**Architecture:**

```
Client Device                          Cloud
+------------------+                   +----------------------------------+
| Microphone       |    WebSocket/     | Load Balancer                    |
| --> VAD          | -- gRPC stream -> | --> ASR Service                  |
| --> Audio Codec  |                   |    --> Feature Extractor          |
| --> Streaming    |                   |    --> Streaming Conformer-RNN-T  |
|    Buffer        |                   |    --> Beam Search Decoder        |
+------------------+                   |    --> Punctuation Model          |
                                       |    --> Inverse Text Norm          |
        <-- Partial results (JSON) --- | --> Response Formatter            |
        <-- Final results ------------ | --> Logging & Analytics           |
                                       +----------------------------------+
```

**Key Design Decisions:**

1. **Streaming Protocol**: WebSocket or gRPC bidirectional streaming
   - gRPC: Lower overhead, built-in protobuf, better for mobile
   - WebSocket: Browser-compatible, simpler to debug

2. **Audio Format**: 16kHz, 16-bit PCM or Opus codec
   - Opus: 10x compression, minimal quality loss
   - Trade-off: Codec latency vs bandwidth savings

3. **Model Architecture**: Conformer-Transducer
   - Conformer encoder (12-17 layers, ~100M params)
   - Stateless prediction network (2 LSTM layers)
   - Joint network
   - Why: Best streaming quality, supports endpoint detection

4. **Decoding Strategy**:
   - Beam search with beam width 4-8
   - Partial (unstable) hypotheses streamed back
   - Endpoint detection triggers final hypothesis
   - Optional: Second-pass rescoring with larger LM

5. **Latency Optimization**:
   - Chunk size: 80-160ms audio chunks
   - Model: INT8 quantization, ~30% faster
   - Batching: Dynamic batching with max 50ms wait
   - Hardware: GPU (NVIDIA T4/A10) with TensorRT

6. **Scaling**:
   - Horizontal pod autoscaler on GPU utilization
   - Regional deployment for latency
   - Session affinity for streaming connections
   - Target: < 300ms end-to-end latency

**Metrics:**
- WER (Word Error Rate): Primary offline metric
- Real-time factor (RTF): Processing time / audio duration, target < 0.3
- P95 latency: < 300ms from audio chunk to partial result
- Availability: 99.95%

---

### Design 2: Voice Assistant / Conversational AI Pipeline

**Problem:** Design an end-to-end voice assistant like Alexa/Siri.

**Architecture:**

```
                     +---> Wake Word Detection (always on, on-device)
                     |          |
Audio Stream --------+    [wake word detected]
                               |
                     +---> ASR (streaming) --> NLU --> Dialog Manager --> NLG --> TTS --> Speaker
                     |                          |           |
                     |                    Intent +    State Tracking
                     |                    Entities     Skill Routing
                     |                                      |
                     +---> Echo Cancellation          +-----+------+
                     +---> Noise Suppression          |     |      |
                                                   Music  Smart  Timer
                                                          Home
```

**Key Components & Decisions:**

1. **Wake Word Detection (On-Device)**
   - Tiny model: depthwise separable CNN, < 500KB
   - Runs continuously on DSP/NPU
   - False accept rate < 1 per 48 hours
   - False reject rate < 3%
   - Training: Large-scale data collection + hard negative mining

2. **Streaming ASR** (see Design 1)

3. **NLU Pipeline**
   - Option A: Traditional -- Intent classifier + NER (slot filling)
   - Option B: Modern -- Fine-tuned LLM with structured output
   - Trade-off: Latency vs flexibility. LLM adds 200-500ms.
   - Hybrid: Fast intent classifier for common intents, LLM fallback for complex

4. **Dialog Management**
   - State machine for simple flows (timers, alarms)
   - LLM-based for open-domain conversation
   - Context window: Last 5-10 turns

5. **TTS** (see dedicated section)
   - Pregenerate common responses
   - Stream TTS output (start speaking before full response generated)
   - Latency target: First audio byte < 500ms after NLU complete

**Latency Budget:**
```
Wake word detection:   ~50ms   (on-device)
Audio streaming:       ~100ms  (network)
ASR:                   ~300ms  (streaming, partial results)
NLU:                   ~100ms  (intent + entities)
Dialog + Skill:        ~200ms  (varies by skill)
TTS first byte:        ~300ms  (streaming synthesis)
-------------------------------------------
Total to first audio:  ~1050ms (target < 1.5s)
```

---

### Design 3: Call Center Speech Analytics

**Problem:** Design a system to analyze millions of customer service calls for quality, compliance, and insights.

**Architecture:**

```
                    Batch Pipeline
Call Recordings --> Audio Preprocessing --> Speaker Diarization --> ASR (per speaker)
  (S3/GCS)         - Resampling            - VAD                   - Whisper large
                   - Normalization          - Embedding extraction  - Domain-adapted
                   - Channel separation     - Clustering

                                    --> Transcript Store (Elasticsearch)
                                            |
                                    +-------+--------+--------+--------+
                                    |       |        |        |        |
                                Sentiment  Compliance  Topic   Agent   Customer
                                Analysis   Detection  Modeling  Score  Satisfaction
                                (per turn) (keyword +  (LDA/   (composite) Prediction
                                           LLM)       BERTopic)
                                    |       |        |        |        |
                                    +-------+--------+--------+--------+
                                            |
                                    --> Analytics Dashboard
                                    --> Alert System (compliance violations)
                                    --> Coaching Recommendations
```

**Key Decisions:**

1. **ASR Choice**: Whisper Large V3 fine-tuned on call center data
   - Why Whisper: Best multilingual, handles noise, good with accents
   - Fine-tune on: Domain vocabulary (product names, jargon)
   - Post-processing: Custom ITN for phone numbers, order IDs

2. **Speaker Diarization**: Critical for call centers (agent vs customer)
   - If stereo audio: Channel separation (trivial)
   - If mono: Pyannote-audio or NeMo diarization
   - Handle overlap: EEND for overlapping speech

3. **Batch vs Streaming**:
   - Batch for historical analysis (cheaper, better quality)
   - Streaming for real-time agent assist (lower quality, higher cost)

4. **Scaling**: 
   - 1M calls/day x 5min avg = ~83K hours of audio/day
   - Whisper Large RTF ~0.05 on A100 = ~4,150 GPU-hours/day
   - Cost optimization: Spot instances, batched processing overnight

---

### Design 4: Real-Time Voice Cloning / Personalized TTS

**Problem:** Design a system that can clone a user's voice from a 10-second sample and generate speech in their voice.

**Architecture:**

```
Enrollment (one-time):
  Reference Audio (10s) --> Speaker Encoder --> Speaker Embedding --> Embedding Store
                            (ECAPA-TDNN)       (256-dim vector)      (Vector DB)

Inference:
  Text Input + Speaker ID --> Fetch Embedding --> TTS Acoustic Model --> Vocoder --> Audio
                                                  (conditioned on         (HiFi-GAN)
                                                   speaker embedding)
```

**Model Options:**

| Approach | Quality | Speed | Data Needed | Example |
|----------|---------|-------|-------------|---------|
| Speaker-conditioned TTS | Good | Fast | 10s-1min | YourTTS, XTTS |
| Codec Language Model | Excellent | Slow | 3-10s | VALL-E, Bark |
| Fine-tuning base TTS | Best | Fast (inference) | 5-30min | Tortoise, Coqui |
| Adapter-based | Good | Fast | 1-5min | LoRA on TTS |

**Key Decisions:**

1. **Zero-shot vs few-shot fine-tuning**
   - Zero-shot (XTTS/VALL-E): Instant, lower quality ceiling
   - Few-shot fine-tune: 5-15 min setup, higher quality
   - Recommendation: Zero-shot for casual, fine-tune for premium

2. **Safety & Ethics** (Tech Lead must address this):
   - Voice consent verification
   - Watermarking generated audio (AudioSeal by Meta)
   - Rate limiting clone requests
   - Deepfake detection integration
   - Legal compliance (right to voice)

---

### Design 5: Multilingual ASR System

**Problem:** Design an ASR system supporting 50+ languages including low-resource languages.

**Key Architecture Decisions:**

1. **Single multilingual model vs per-language models**
   - Single model: Easier to maintain, cross-lingual transfer
   - Per-language: Better for high-resource, more complex ops
   - Recommendation: Multilingual base + language-specific adapters

2. **Language Identification**:
   ```
   Audio --> LID Model (first 2s) --> Route to language-specific adapter
   ```
   Or: Jointly model LID + ASR (like Whisper)

3. **Pre-training Strategy**:
   - Start from Whisper or MMS (Meta's Massively Multilingual Speech)
   - MMS: 1100+ languages, based on Wav2Vec 2.0
   - Fine-tune on target language data

4. **Low-Resource Language Tricks**:
   - Cross-lingual transfer from related high-resource language
   - Self-training: Use teacher model to generate pseudo-labels
   - Data augmentation: Speed perturbation, SpecAugment, noise injection
   - Use phoneme-based output (IPA) for very low-resource

---

### Design 6: Voice-Based Emotion/Sentiment Detection

**Problem:** Detect speaker emotions in real-time during customer calls.

**Architecture:**

```
Audio Stream --> Feature Extraction --> Emotion Model --> Temporal Smoothing --> Output
                 - Mel spectrogram      - HuBERT/WavLM     - Running avg       - Emotion label
                 - Prosodic features    - Fine-tuned        - HMM               - Confidence
                 - Energy, pitch        - Multi-task head   - Majority vote     - Valence/Arousal
```

**Key Decisions:**

1. **Categorical vs Dimensional**:
   - Categorical: Happy, Sad, Angry, Neutral, Frustrated (easier to act on)
   - Dimensional: Valence + Arousal continuous values (more nuanced)
   - Recommendation: Both -- dimensional internally, mapped to categorical for UI

2. **Text vs Audio vs Multimodal**:
   - Audio-only: Captures tone, independent of language
   - Text-only (from ASR): Captures semantic sentiment
   - Multimodal (both): Best accuracy, higher latency
   - Recommendation: Late fusion of audio + text models

3. **Pre-trained backbone**: WavLM or Emotion2Vec
   - Fine-tune classification head on labeled emotion data
   - Multi-task with ASR to regularize

---

### Design 7: Audio Deepfake Detection System

### Design 8: Voice Search System (E-commerce/Music)

### Design 9: Real-Time Translation Pipeline (Speech-to-Speech)

### Design 10: Audio Content Moderation at Scale

*(Designs 7-10 follow similar frameworks -- the key is applying the same structured approach)*

---

## 4. Architecture Patterns & Building Blocks

### Pattern 1: Streaming vs Batch

| Aspect | Streaming | Batch |
|--------|-----------|-------|
| Latency | < 500ms | Minutes to hours |
| Quality | Lower (partial context) | Higher (full context) |
| Cost | Higher (always-on GPU) | Lower (spot instances) |
| Complexity | Higher (state management) | Lower |
| Use cases | Real-time ASR, live captioning | Call analytics, training |

**Trick:** "I'd implement a dual-mode architecture: streaming for real-time user experience, with batch re-processing for higher-quality transcripts stored asynchronously."

### Pattern 2: Two-Pass Architecture

```
Pass 1 (Streaming): Small fast model --> Partial results to user (low latency)
Pass 2 (Batch): Large accurate model --> Corrected final results (high quality)
```

Used by: Google ASR, YouTube captions. Shows you think about user experience + quality.

### Pattern 3: Cascade vs End-to-End

**Cascade (Traditional):**
```
ASR --> NLU --> Dialog --> NLG --> TTS
```
- Pros: Modular, debuggable, each component optimizable
- Cons: Error propagation, higher latency, information loss

**End-to-End (Modern):**
```
Audio --> Single Large Model --> Audio/Text Response
```
- Pros: Lower latency, no error propagation, captures nuance
- Cons: Harder to debug, less controllable, needs massive data

**Trick:** "For production, I'd start with a cascade architecture for debuggability and controllability, then explore E2E for latency-critical paths once we have sufficient data."

### Pattern 4: On-Device + Cloud Hybrid

```
On-Device:                          Cloud:
- Wake word detection               - Full ASR
- VAD                               - NLU
- Small ASR (for common commands)   - Dialog management
- Audio preprocessing                - TTS (complex)
- Keyword spotting                   - Analytics

Decision logic: If confidence > threshold, use on-device result
                Else, send to cloud
```

### Pattern 5: Teacher-Student / Distillation

```
Large Teacher Model (Whisper Large, offline) --> Soft labels
                                                     |
Student Model (small, streaming) <-- Train on soft labels + hard labels
```

Use this to explain how you'd compress models for production.

---

## 5. Key Metrics & Evaluation

### ASR Metrics

| Metric | Formula | Target | Notes |
|--------|---------|--------|-------|
| WER | (S+D+I)/N | < 5% | Primary metric |
| CER | Character-level WER | < 3% | Better for CJK languages |
| RTF | Process time / Audio duration | < 0.3 | Speed metric |
| Latency P50 | 50th percentile response time | < 200ms | User experience |
| Latency P95 | 95th percentile | < 500ms | Tail latency |
| Endpointing Delay | Time after speech ends to final result | < 500ms | Conversational feel |

### TTS Metrics

| Metric | Type | Notes |
|--------|------|-------|
| MOS (Mean Opinion Score) | Human eval | Gold standard, 1-5 scale |
| PESQ | Objective | Perceptual quality |
| UTMOS | Predicted MOS | Neural network predicts MOS |
| Speaker Similarity | Cosine similarity of embeddings | For voice cloning |
| RTF | Processing speed | Must be < 1.0 for real-time |

### Business Metrics (Tech Lead Must Discuss)

| Metric | What It Measures |
|--------|-----------------|
| Task completion rate | % of user requests successfully fulfilled |
| User retention | % of users who return |
| Escalation rate | % of calls transferred to human agent |
| Average handle time (AHT) | Duration of customer interactions |
| Customer satisfaction (CSAT) | Post-interaction survey scores |
| Cost per interaction | Infrastructure + API costs per conversation |

---

## 6. Scaling & Production

### Model Optimization Toolkit

| Technique | Speedup | Quality Loss | Effort |
|-----------|---------|-------------|--------|
| FP16 inference | 1.5-2x | Negligible | Low |
| INT8 quantization (PTQ) | 2-3x | 0.1-0.5% WER | Low |
| INT8 quantization (QAT) | 2-3x | < 0.1% WER | Medium |
| INT4 quantization | 3-5x | 0.5-2% WER | Medium |
| Pruning (structured) | 1.5-3x | 0.2-1% WER | Medium |
| Knowledge distillation | Model-dependent | 0.5-2% WER | High |
| TensorRT / ONNX Runtime | 1.5-3x | None | Low-Med |
| Speculative decoding | 2-3x | None | Medium |
| Flash Attention | 1.5-2x | None | Low |

### Serving Infrastructure

```
                    +---> GPU Pool (ASR inference)
                    |     - NVIDIA Triton Inference Server
                    |     - Dynamic batching
                    |     - Model versioning
API Gateway -->  Load Balancer
                    |     
                    +---> GPU Pool (TTS inference)
                    |     - Streaming audio output
                    |     - Voice-specific model routing
                    |
                    +---> CPU Pool (Pre/Post processing)
                          - Feature extraction
                          - Text normalization
                          - Punctuation restoration
```

**Key Infrastructure Choices:**

| Decision | Recommendation | Why |
|----------|---------------|-----|
| Inference server | NVIDIA Triton | Multi-framework, dynamic batching, model ensemble |
| Orchestration | Kubernetes + GPU operator | Auto-scaling, resource management |
| Streaming | gRPC bidirectional | Low overhead, native streaming |
| Model format | ONNX or TensorRT | Optimized inference |
| Caching | Redis for embeddings | Avoid recomputing speaker embeddings |
| Queue | Kafka for batch jobs | Reliable, high-throughput |

### Monitoring & Observability

```
Real-time Monitoring:
  - Request latency (P50, P95, P99)
  - GPU utilization & memory
  - Error rates by error type
  - Queue depth
  - Concurrent connections

ML-Specific Monitoring:
  - WER on shadow test set (daily)
  - Confidence score distribution drift
  - Audio quality metrics (SNR) of inputs
  - Out-of-vocabulary (OOV) rate
  - Language/accent distribution shifts

Alerting:
  - WER regression > 1% absolute
  - P95 latency > SLA threshold
  - GPU memory > 90%
  - Error rate > 0.1%
```

### Data Flywheel (Critical for Tech Lead)

```
Users --> Audio --> ASR --> Transcripts
                    |           |
                    v           v
            Confidence     Human Review
            Filtering      (low confidence)
                    |           |
                    v           v
            High-confidence   Corrected
            Auto-labels       Labels
                    |           |
                    +-----+-----+
                          |
                    Model Retraining
                          |
                    A/B Test --> Ship
```

**This is your competitive moat.** More users -> more data -> better model -> more users.

---

## 7. Tech Lead Specific: Leadership Signals

### What Interviewers Look For in Tech Lead Candidates

1. **System thinking**: Can you see the whole picture, not just the ML model?
2. **Trade-off articulation**: Can you explain WHY, not just WHAT?
3. **Prioritization**: What do you build first vs later?
4. **Risk assessment**: What can go wrong and how do you mitigate?
5. **Team & process**: How would you organize the team to build this?
6. **Business alignment**: How does this system drive business value?

### Phrases That Signal Tech Lead Thinking

Use these naturally in your answers:

- "The first thing I'd want to understand is the business objective behind this..."
- "Given the constraint of [X], I'd prioritize [Y] over [Z] because..."
- "The main risk here is [X]. To mitigate that, we'd..."
- "For the MVP, I'd scope this down to [X]. In V2, we'd add [Y]..."
- "I'd set up the team in [X] way, with [role] owning [component]..."
- "The data flywheel here is critical -- here's how we'd set it up..."
- "Before building, I'd want to validate our assumptions with [experiment]..."
- "The key metric we'd optimize for is [X], with guardrails on [Y]..."
- "This is a reversible/irreversible decision, so I'd [move fast/be more careful]..."

### How to Discuss Team Structure

```
Voice AI Team (12 people):
  - 2 Research Scientists: Model architecture, pre-training
  - 3 ML Engineers: Training infra, fine-tuning, evaluation
  - 2 Data Engineers: Audio pipelines, data quality, labeling
  - 2 Backend Engineers: Serving, APIs, scaling
  - 1 Audio/DSP Engineer: Signal processing, feature extraction
  - 1 Applied Scientist: A/B testing, metrics, analysis
  - 1 Tech Lead (you): Architecture, roadmap, cross-team coordination
```

### MVP vs Full System Thinking

Always present a phased approach:

**Phase 1 (MVP, 4-6 weeks):**
- Use off-the-shelf ASR (Whisper API / Google Cloud STT)
- Simple rule-based post-processing
- Basic monitoring
- Validate the business case

**Phase 2 (Custom, 3-6 months):**
- Train domain-adapted ASR model
- Custom post-processing pipeline
- A/B testing infrastructure
- Data flywheel setup

**Phase 3 (Differentiated, 6-12 months):**
- Fully custom E2E model
- On-device inference for latency-critical paths
- Multi-language support
- Advanced features (emotion, speaker ID)

---

## 8. Speed-Run Preparation Plan

### 2-Week Intensive Plan

#### Week 1: Foundations & Framework

| Day | Focus | Activities |
|-----|-------|-----------|
| 1 | Framework | Memorize the 45-min structure. Practice on 2 generic ML system design problems. |
| 2 | ASR Deep Dive | Study Conformer, RNN-T, CTC. Draw architectures from memory. |
| 3 | ASR Deep Dive | Study Whisper, Wav2Vec2, HuBERT. Understand pre-training approaches. |
| 4 | TTS Deep Dive | Study FastSpeech2, VITS, HiFi-GAN. Understand the acoustic model + vocoder pipeline. |
| 5 | Practice | Design "Real-time ASR" and "Voice Assistant" end-to-end. Time yourself (45 min each). |
| 6 | Supporting Tech | Study VAD, speaker diarization, speaker verification, noise suppression. |
| 7 | Review | Re-read notes. Focus on weak areas. Practice explaining trade-offs out loud. |

#### Week 2: Practice & Polish

| Day | Focus | Activities |
|-----|-------|-----------|
| 8 | System Design | Design "Call Center Analytics" and "Voice Cloning" systems. |
| 9 | Production | Study serving, quantization, monitoring. Practice explaining scaling decisions. |
| 10 | Mock Interview | Do a full 45-min mock with a friend or record yourself. Review recording. |
| 11 | Edge Cases | Study: multilingual ASR, emotion detection, deepfake detection. |
| 12 | Mock Interview | Second mock interview. Focus on incorporating feedback from Day 10. |
| 13 | Leadership | Practice tech lead signals: team structure, prioritization, phased roadmaps. |
| 14 | Final Review | Review all designs. Create 1-page cheat sheet. Rest and prepare mentally. |

### Key Resources (Prioritized)

**Must-Read Papers (Read abstracts + key results):**
1. "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
2. "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision" (2022)
3. "Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
4. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)
5. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
6. "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023)

**Must-Watch (If you prefer video):**
- Stanford CS224S: Spoken Language Processing
- Hung-yi Lee's Speech Processing lectures (YouTube, free)
- NVIDIA GTC talks on ASR deployment

**Must-Practice:**
- Design 3 full systems end-to-end (ASR, Voice Assistant, Call Analytics)
- Practice explaining each design in 45 minutes with a timer

### Speed Tips (Get Results Faster)

1. **Template everything**: Have your framework memorized so you spend zero time on structure
2. **Default choices ready**: Know your go-to model for each component (see cheat sheet below)
3. **Practice out loud**: ML system design is a verbal exam. Reading is not enough.
4. **Record yourself**: Watch the recording. You'll catch unclear explanations immediately.
5. **Focus on trade-offs**: Interviewers care more about WHY than WHAT. Every design decision should have a "because..."
6. **Draw first, talk second**: Drawing the architecture gives you thinking time
7. **Anchor on numbers**: "Whisper Large has 1.5B params and achieves 2.7% WER on LibriSpeech" -- shows you know the field
8. **Don't get stuck on one component**: If the interviewer seems bored, move on. Depth on 2-3 components, breadth on everything.

---

## 9. Cheat Sheets

### Go-To Defaults (When in Doubt, Pick These)

| Component | Default Choice | Why |
|-----------|---------------|-----|
| Streaming ASR | Conformer-Transducer | Best streaming quality |
| Batch ASR | Whisper Large V3 | Best off-the-shelf, multilingual |
| TTS Acoustic | VITS / XTTS | End-to-end, good quality |
| Vocoder | HiFi-GAN | Fast, high quality |
| Voice Cloning | XTTS v2 | Good zero-shot, multilingual |
| Speaker Embedding | ECAPA-TDNN | State-of-art speaker verification |
| VAD | Silero VAD | Fast, accurate, open source |
| Diarization | Pyannote-audio 3.0 | Best open-source pipeline |
| Pre-trained backbone | WavLM Large | Robust, good for downstream tasks |
| Noise Suppression | DTLN / RNNoise | Real-time capable |
| Inference Server | NVIDIA Triton | Dynamic batching, multi-model |
| Audio Codec | Opus (transmission), EnCodec (ML) | Standard choices |
| Feature | 80-dim log-mel filterbank | Universal audio feature |

### Numbers You Should Know

| Metric | Value |
|--------|-------|
| Human WER (clean speech) | ~4% |
| Whisper Large V3 WER (LibriSpeech test-clean) | ~2.7% |
| Conformer-T WER (LibriSpeech test-clean) | ~1.9% |
| Typical conversational speech rate | 120-150 words/min |
| Typical audio sample rate for ASR | 16 kHz |
| Typical audio sample rate for TTS | 22.05 or 24 kHz |
| Log-mel filterbank dimensions | 80 |
| Mel spectrogram frame size | 25ms window, 10ms hop |
| Whisper Large model size | ~1.5B parameters, ~3GB |
| Conformer-L model size | ~120M parameters, ~500MB |
| Acceptable latency for conversational AI | < 500ms to first token |
| GPU memory for Whisper Large (FP16) | ~3GB |
| Cost of Whisper Large on A100 (RTF) | ~0.03 (33x real-time) |
| MOS score for natural speech | ~4.5 |
| MOS score for best TTS | ~4.2-4.4 |

### Common Pitfalls to Avoid

1. **Jumping to model architecture without clarifying requirements** -- Always start with requirements
2. **Ignoring data** -- The model is 20% of the system. Data pipelines are 80%.
3. **Not discussing latency** -- For voice, latency IS the product
4. **Forgetting about errors** -- What happens when ASR is wrong? What's the fallback?
5. **Over-engineering the MVP** -- Start simple, iterate
6. **Ignoring cost** -- "I'd use 100 A100s" is not a realistic answer
7. **Not quantifying** -- Use numbers. "About 10M users" is better than "many users"
8. **Monologue-ing** -- Check in with the interviewer. "Would you like me to go deeper on X or move to Y?"

---

## 10. Mock Interview Scripts

### Script 1: "Design a real-time transcription service for video meetings"

**Your answer structure:**

> "Great question. Let me start by clarifying a few things...
>
> **Clarify:** How many concurrent meetings? (Let's say 100K). Languages? (English first, then multilingual). Do we need speaker labels? (Yes). Real-time captions or post-meeting transcript? (Both). What's our latency budget? (< 1s for captions).
>
> **High-level design:** [Draw the pipeline]
> Audio streams from clients --> WebSocket --> Audio Router --> Streaming ASR --> Speaker Diarization --> Post-processing --> Caption Stream back + Stored Transcript
>
> **Deep dive on ASR:**
> I'd use a Conformer-Transducer architecture because we need streaming with good quality. The encoder would be a 12-layer Conformer, and we'd use RNN-T for the decoder to get streaming output. For the meeting domain, we'd fine-tune on meeting-style speech with overlapping speakers, filler words, and domain-specific vocabulary.
>
> **Deep dive on Diarization:**
> For real-time, I'd use a turn-based approach: detect speaker changes using speaker embedding comparison. Extract x-vectors every 1s window, compare with known speakers using cosine similarity. When similarity drops below threshold, start a new speaker.
>
> **Scaling:**
> 100K concurrent meetings x 2-8 speakers = up to 800K audio streams. With dynamic batching on GPUs, each A10 GPU can handle ~50 concurrent streams. So ~16K GPUs. That's expensive -- let me think about optimization...
>
> We could: (1) Use INT8 quantization to double throughput, (2) Only run ASR when speech is detected (VAD), which cuts compute by 40-60%, (3) Use smaller model for meetings with < 4 speakers.
>
> **Trade-offs:** I chose server-side ASR over on-device because meeting audio often has multiple speakers and we need cross-speaker context. But we could do a hybrid: VAD on-device, ASR in cloud.
>
> **Metrics:** WER < 10% for meeting speech (harder than clean speech), caption latency P95 < 1s, speaker identification accuracy > 90%.
>
> **Extensions:** Add real-time translation, meeting summarization using LLM on transcripts, action item extraction."

---

## Appendix A: Voice AI Glossary (Quick Reference)

| Term | Definition |
|------|-----------|
| WER | Word Error Rate = (Substitutions + Deletions + Insertions) / Total Words |
| CER | Character Error Rate (same formula, character level) |
| RTF | Real-Time Factor = processing time / audio duration (< 1 means faster than real-time) |
| VAD | Voice Activity Detection -- detecting when someone is speaking |
| ASR | Automatic Speech Recognition (speech-to-text) |
| TTS | Text-to-Speech |
| STT | Speech-to-Text (same as ASR) |
| S2S | Speech-to-Speech |
| NLU | Natural Language Understanding |
| NLG | Natural Language Generation |
| CTC | Connectionist Temporal Classification (alignment-free loss function) |
| RNN-T | RNN-Transducer (streaming E2E ASR architecture) |
| LAS | Listen, Attend and Spell (attention-based ASR) |
| AED | Attention-based Encoder-Decoder |
| ITN | Inverse Text Normalization (converting "five dollars" to "$5") |
| G2P | Grapheme-to-Phoneme (text to pronunciation) |
| MOS | Mean Opinion Score (human TTS quality rating, 1-5) |
| PESQ | Perceptual Evaluation of Speech Quality |
| MFCC | Mel-Frequency Cepstral Coefficients (audio features) |
| Mel Spectrogram | Time-frequency representation weighted by human hearing |
| Endpointing | Detecting when a user has finished speaking |
| Diarization | Determining "who spoke when" in multi-speaker audio |
| Beamforming | Using multiple mics to focus on a sound source |
| AEC | Acoustic Echo Cancellation |
| AGC | Automatic Gain Control |
| SNR | Signal-to-Noise Ratio |
| EnCodec | Neural audio codec by Meta (discrete audio tokens) |
| SoundStream | Neural audio codec by Google |

## Appendix B: System Design Diagram Templates

### Template 1: Generic Voice AI System
```
+--------+     +--------+     +---------+     +--------+     +--------+
| Audio  | --> | Pre-   | --> | Feature | --> | Model  | --> | Post-  | --> Output
| Input  |     | process|     | Extract |     | Infer  |     | process|
+--------+     +--------+     +---------+     +--------+     +--------+
                   |                               |              |
              Noise/Echo                      GPU Cluster     Beam Search
              VAD                             Batching        ITN/Punct
              Resampling                      Caching         Formatting
```

### Template 2: Training Pipeline
```
Raw Audio --> Data Pipeline --> Feature Store --> Training Loop --> Evaluation --> Registry
              - Augmentation    - Mel specs       - Distributed      - WER         - Model
              - Filtering       - Labels          - Mixed precision  - Latency     - Version
              - Splitting       - Metadata        - Checkpointing   - A/B test    - Deploy
```

---

*This guide is designed for rapid preparation. Focus on understanding the frameworks and being able to explain trade-offs clearly. The best candidates don't memorize -- they reason through problems using solid fundamentals.*

*Good luck with your interview!*
