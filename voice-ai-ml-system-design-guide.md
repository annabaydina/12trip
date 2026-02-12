# ML System Design for Voice AI Engineer Tech Lead -- Complete Preparation Guide

## Table of Contents

1. [The Interview Framework -- Your 45-Minute Blueprint](#1-the-interview-framework)
2. [Voice AI Domain Deep Dive](#2-voice-ai-domain-deep-dive)
3. [Core ML System Design Template](#3-core-ml-system-design-template)
4. [Voice AI Architecture Patterns](#4-voice-ai-architecture-patterns)
5. [Most Likely Interview Questions & How to Crush Them](#5-most-likely-interview-questions)
6. [Latency & Real-Time Constraints -- The Differentiator](#6-latency-and-real-time-constraints)
7. [Evaluation Metrics That Impress](#7-evaluation-metrics)
8. [Production & MLOps for Voice Systems](#8-production-and-mlops)
9. [Leadership & Tech Lead Angles](#9-leadership-and-tech-lead-angles)
10. [Speed-Run Study Plan (2 Weeks)](#10-speed-run-study-plan)
11. [Resources & References](#11-resources-and-references)

---

## 1. The Interview Framework -- Your 45-Minute Blueprint <a name="1-the-interview-framework"></a>

The single biggest mistake candidates make is diving into model architecture. Follow this structured approach instead:

### Time Allocation

| Phase | Time | What to Cover |
|-------|------|---------------|
| **Clarify Requirements** | 5 min | Functional/non-functional requirements, constraints |
| **High-Level Design** | 10 min | End-to-end pipeline, component diagram |
| **Deep Dive** | 15 min | Model choices, data pipeline, feature engineering |
| **Scaling & Production** | 10 min | Serving, latency, throughput, fault tolerance |
| **Evaluation & Iteration** | 5 min | Metrics, A/B testing, monitoring |

### The Golden Rules

1. **Always start with "What problem are we solving?"** -- not "What model should we use?"
2. **Quantify everything** -- "We need < 200ms p99 latency" not "it should be fast"
3. **Trade-offs are king** -- every decision should come with a trade-off discussion
4. **Think like a Tech Lead** -- mention team structure, timelines, build-vs-buy decisions

---

## 2. Voice AI Domain Deep Dive <a name="2-voice-ai-domain-deep-dive"></a>

### The Voice AI Stack (Know This Cold)

```
User Speaks
    |
    v
[Audio Capture] --> [VAD] --> [ASR/STT] --> [NLU/Intent] --> [Dialog Manager] --> [NLG] --> [TTS] --> Speaker
                     |            |              |                |                          |
                  Voice        Speech-to-     Natural          Conversation              Text-to-
                  Activity     Text           Language         State Machine              Speech
                  Detection                   Understanding
```

### Key Components You MUST Know

#### 2.1 Automatic Speech Recognition (ASR / STT)

**Traditional Pipeline:**
- Acoustic Model (AM): Audio features -> phonemes
- Language Model (LM): Phoneme sequences -> word sequences
- Decoder: Beam search / CTC decoding

**Modern End-to-End:**
- **Whisper (OpenAI)**: Encoder-decoder transformer, multilingual, robust to noise
- **Conformer**: Convolution + Transformer hybrid, state-of-the-art for streaming
- **RNN-Transducer (RNN-T)**: Best for streaming/real-time (used by Google, Meta)
- **CTC-based models**: wav2vec 2.0, HuBERT -- great for low-resource + fine-tuning

**Key Trade-off to discuss:**
- Streaming vs. batch ASR
  - Streaming (RNN-T, Conformer): Lower latency, slightly lower accuracy
  - Batch (Whisper): Higher accuracy, higher latency
  - For real-time voice AI: streaming is almost always required

#### 2.2 Text-to-Speech (TTS)

**Modern Architecture:**
```
Text --> [Text Encoder] --> [Acoustic Model] --> [Mel Spectrogram] --> [Vocoder] --> Audio
```

**Models to know:**
- **VITS / VITS2**: End-to-end, high quality, single model
- **Tacotron 2 + HiFi-GAN**: Classic pipeline, well-understood
- **FastSpeech 2 + vocoder**: Non-autoregressive, faster inference
- **VALL-E / Bark / XTTS**: Zero-shot voice cloning via LLM-style generation
- **Tortoise TTS**: High quality but slow (not for real-time)

**Key Trade-off:**
- Quality vs. latency vs. voice cloning capability
- Autoregressive (higher quality, slower) vs. non-autoregressive (faster, sometimes less natural)

#### 2.3 Voice Activity Detection (VAD)

- **Silero VAD**: Lightweight, production-ready, works everywhere
- **WebRTC VAD**: Ultra lightweight, built into browsers
- **Custom CNN/RNN**: When you need domain-specific tuning

**Why VAD matters for system design:**
- Reduces compute by only processing speech segments
- Critical for turn-taking in conversational AI
- Endpointing: detecting when user finished speaking

#### 2.4 Speaker Verification / Identification

- **d-vectors, x-vectors**: Traditional embedding approaches
- **ECAPA-TDNN**: Current SOTA for speaker verification
- **Resemblyzer**: Quick prototyping
- **TitaNet (NVIDIA)**: Production-grade

#### 2.5 Natural Language Understanding (NLU)

- **Intent classification**: BERT/DistilBERT fine-tuned
- **Slot filling / NER**: Token classification
- **LLM-based**: GPT-4, Claude, Llama for complex dialog understanding
- **Joint models**: Intent + slots in single model (JointBERT)

---

## 3. Core ML System Design Template <a name="3-core-ml-system-design-template"></a>

Use this template for EVERY voice AI system design question:

### Step 1: Requirements Gathering (Ask These Questions)

**Functional:**
- What languages/accents must be supported?
- Real-time or batch processing?
- Is voice cloning/customization needed?
- What's the dialog complexity (simple commands vs. open-ended)?
- On-device or cloud?

**Non-Functional:**
- Latency budget (end-to-end, per component)?
- Throughput (concurrent users)?
- Accuracy targets (WER for ASR, MOS for TTS)?
- Privacy/compliance requirements (GDPR, HIPAA)?
- Cost constraints?

### Step 2: High-Level Architecture

Always draw these components:
1. **Client/Edge** -- audio capture, local VAD, audio codec
2. **API Gateway** -- authentication, rate limiting, routing
3. **Streaming Infrastructure** -- WebSocket / gRPC for real-time audio
4. **ML Pipeline** -- ASR -> NLU -> Dialog -> NLG -> TTS
5. **Data Pipeline** -- logging, feedback collection, training data
6. **Model Registry & Serving** -- versioning, A/B testing, canary deploys

### Step 3: Data Strategy

```
Raw Audio --> [Preprocessing] --> [Feature Extraction] --> [Training Data Store]
                  |                      |
            - Resampling (16kHz)    - Mel spectrograms
            - Noise reduction       - MFCCs
            - Normalization         - Filter banks
            - Diarization           - wav2vec embeddings
```

**Data considerations for voice AI:**
- Audio data is LARGE (1 hour of 16kHz mono = ~115 MB uncompressed)
- Transcription is expensive ($1-2/hour for human annotation)
- Synthetic data generation (TTS for ASR training, speed/pitch perturbation)
- Data augmentation: noise injection, room impulse response, speed perturbation
- Privacy: PII in audio, voice biometrics are sensitive data

### Step 4: Model Selection Decision Framework

```
                        Need real-time?
                       /              \
                     Yes               No
                    /                    \
            Streaming ASR            Batch ASR
           (RNN-T, Conformer)       (Whisper large)
                |                        |
         On-device possible?      Accuracy critical?
            /        \               /        \
          Yes         No           Yes         No
           |           |            |           |
      Quantized    Cloud-based   Ensemble    Single model
      small model   full model   + LM rescore
```

### Step 5: Serving Architecture

**For real-time voice AI (most common scenario):**

```
Client <--WebSocket--> Load Balancer <--> ASR Service (GPU)
                                    <--> NLU Service (CPU/GPU)
                                    <--> Dialog Service (CPU)
                                    <--> TTS Service (GPU)
```

**Key serving decisions:**
- **gRPC streaming** for inter-service communication (lower overhead than REST)
- **GPU sharing**: ASR and TTS on same GPU with time-multiplexing
- **Batching**: Dynamic batching for throughput (NVIDIA Triton)
- **Model format**: ONNX Runtime, TensorRT, or OpenVINO for inference optimization

---

## 4. Voice AI Architecture Patterns <a name="4-voice-ai-architecture-patterns"></a>

### Pattern 1: Cascaded Pipeline (Traditional)

```
Audio -> VAD -> ASR -> NLU -> Dialog Manager -> NLG -> TTS -> Audio
```

**Pros:** Modular, each component independently optimizable, easier debugging
**Cons:** Error propagation, higher total latency, integration complexity

**When to use:** Enterprise voice assistants, IVR systems, when you need fine-grained control

### Pattern 2: End-to-End with LLM (Modern)

```
Audio -> ASR -> LLM (understands + generates) -> TTS -> Audio
```

**Pros:** Simpler pipeline, LLM handles NLU + dialog + NLG, more natural conversations
**Cons:** LLM latency, cost, less control over specific behaviors

**When to use:** Conversational AI assistants, when naturalness matters most

### Pattern 3: Speech-to-Speech (Cutting Edge)

```
Audio -> [Single Model: understands speech, generates speech tokens] -> Audio
```

**Examples:** AudioPaLM, SpeechGPT, VITA, Moshi
**Pros:** Lowest latency potential, preserves prosody/emotion
**Cons:** Still experimental, harder to control, less interpretable

**When to use:** Next-gen voice assistants, when you want to show you know the frontier

### Pattern 4: Hybrid Edge-Cloud

```
[Edge Device]                          [Cloud]
Audio -> VAD -> Streaming ASR -------> Full ASR (refinement)
         |                              |
    Wake word detection            NLU + Dialog + NLG
         |                              |
    Local TTS (simple) <----------  Cloud TTS (complex)
```

**When to use:** Smart speakers, automotive, IoT devices

---

## 5. Most Likely Interview Questions & How to Crush Them <a name="5-most-likely-interview-questions"></a>

### Q1: "Design a real-time voice assistant (like Alexa/Siri)"

**Your answer structure:**

1. **Clarify**: Always-on? Multi-language? On-device wake word?
2. **Wake Word Detection**: Small CNN on device (< 200KB), always listening
3. **Streaming ASR**: Conformer/RNN-T, streaming via WebSocket
4. **NLU**: Intent classifier + slot filler, or LLM for complex queries
5. **Dialog Management**: State machine for simple, LLM for complex
6. **TTS**: FastSpeech2 + HiFi-GAN for low latency, cache common responses
7. **Latency budget breakdown**:
   - Wake word: 0ms (always running)
   - Audio streaming: 100ms (network)
   - ASR (streaming): 200-300ms after end of speech
   - NLU + Dialog: 50-100ms
   - TTS: 200-300ms (first chunk, then streaming)
   - **Total: ~600ms-800ms** first audio response

### Q2: "Design a speech-to-text system for medical dictation"

**Key angles:**
- **Accuracy is paramount**: Medical terminology, drug names
- **Custom language model**: Domain-specific n-gram LM for rescoring
- **Vocabulary**: Expandable medical vocabulary, UMLS integration
- **Privacy**: HIPAA compliance, on-premise deployment option
- **Correction UX**: Confidence scores, easy correction interface
- **Speaker adaptation**: Fine-tune acoustic model per doctor
- **Structured output**: Auto-format into SOAP notes

### Q3: "Design a voice cloning system"

**Architecture:**
```
Enrollment: Reference audio (5-30s) -> Speaker Encoder -> Speaker Embedding
                                                              |
Synthesis: Text -> Text Encoder -> Decoder(+speaker embed) -> Mel -> Vocoder -> Cloned Audio
```

**Key points:**
- **Few-shot vs. fine-tuned**: VALL-E style (few-shot) vs. fine-tuning (higher quality)
- **Ethics & safety**: Deepfake detection, consent verification, watermarking
- **Evaluation**: MOS (Mean Opinion Score), speaker similarity (cosine similarity of embeddings)

### Q4: "Design a real-time translation system (speech-to-speech)"

**Architecture:**
```
Source Audio -> ASR (source lang) -> MT (translation) -> TTS (target lang) -> Target Audio
   OR
Source Audio -> S2T (speech-to-text translation, e.g., SeamlessM4T) -> TTS -> Target Audio
   OR
Source Audio -> S2S (end-to-end, e.g., Translatotron 2) -> Target Audio
```

**Trade-offs to discuss:**
- Cascaded (more accurate per component) vs. end-to-end (lower latency, preserves voice)
- Simultaneous translation vs. sentence-level
- SeamlessM4T from Meta as a strong baseline

### Q5: "Design a call center voice AI system"

**This is a GREAT question for Tech Lead candidates. Cover:**
1. **IVR integration**: SIP/telephony, 8kHz audio handling
2. **Real-time ASR**: Handle accents, noise, telephony compression artifacts
3. **Intent routing**: Classify and route to right department/bot
4. **Agent assist**: Real-time transcription + suggestions for human agents
5. **Analytics pipeline**: Call sentiment, topic extraction, compliance monitoring
6. **Scale**: 10K+ concurrent calls, geographic distribution
7. **Fallback**: Graceful handoff to human agents

---

## 6. Latency & Real-Time Constraints -- The Differentiator <a name="6-latency-and-real-time-constraints"></a>

This is where Voice AI system design differs most from other ML system design. **Master this section.**

### Latency Budget Template

| Component | Target | Optimization |
|-----------|--------|-------------|
| Audio capture + encoding | 20-40ms | Opus codec, chunk size tuning |
| Network (client -> server) | 20-50ms | Edge deployment, WebSocket |
| VAD | 5-10ms | Silero VAD, run on CPU |
| ASR (streaming) | 100-300ms | RNN-T, Conformer with chunked attention |
| NLU/Intent | 10-50ms | DistilBERT, ONNX optimized |
| LLM (if used) | 200-500ms TTFT | Speculative decoding, KV cache, smaller models |
| TTS (first chunk) | 50-200ms | Non-autoregressive, chunk-based streaming |
| Network (server -> client) | 20-50ms | Audio streaming, jitter buffer |
| **Total end-to-end** | **< 1000ms** | Budget carefully per component |

### Optimization Techniques (Mention These!)

1. **Model optimization:**
   - Quantization: INT8/INT4 (2-4x speedup, minimal accuracy loss)
   - Pruning: Remove redundant weights (20-50% reduction)
   - Knowledge distillation: Train small model from large model
   - ONNX Runtime / TensorRT: Hardware-optimized inference

2. **System optimization:**
   - Streaming pipeline: Start processing before user finishes speaking
   - Speculative execution: Start TTS while LLM is still generating
   - Audio chunking: Process 20-40ms chunks, don't wait for full utterance
   - Connection pooling: Keep WebSocket/gRPC connections alive
   - GPU memory management: Pre-allocate, avoid dynamic allocation

3. **Caching:**
   - TTS audio caching for common responses ("I didn't catch that", "How can I help?")
   - ASR language model caching (frequently used phrases)
   - Embedding caches for speaker verification

4. **Architecture-level:**
   - Prefetch: Start loading TTS model while ASR is running
   - Pipeline parallelism: Overlap compute across pipeline stages
   - Adaptive quality: Reduce model size under high load

---

## 7. Evaluation Metrics That Impress <a name="7-evaluation-metrics"></a>

### ASR Metrics
- **WER (Word Error Rate)**: Primary metric, lower is better
- **CER (Character Error Rate)**: Better for morphologically rich languages
- **RTF (Real-Time Factor)**: Processing time / audio duration (must be < 1 for real-time)
- **Latency**: Time from end of speech to transcript availability

### TTS Metrics
- **MOS (Mean Opinion Score)**: Human evaluation, 1-5 scale (target > 4.0)
- **PESQ/POLQA**: Automated speech quality metrics
- **Speaker similarity**: Cosine similarity of speaker embeddings
- **RTF**: Must be < 1 for real-time, ideally < 0.1

### System-Level Metrics
- **Task completion rate**: Did the user accomplish their goal?
- **Turn count**: Fewer turns = better understanding
- **User satisfaction (CSAT)**: Survey-based
- **Fallback rate**: How often does the system fail to understand?
- **End-to-end latency**: p50, p95, p99
- **Concurrent session capacity**: At what load does quality degrade?

### A/B Testing for Voice AI (Important for Tech Lead)
- **Interleaved experiments**: Route % of traffic to new model
- **Stratified by**: Language, accent, device type, noise level
- **Watch for**: Latency regression, WER per demographic group
- **Statistical significance**: Voice AI needs more samples (high variance in audio)

---

## 8. Production & MLOps for Voice Systems <a name="8-production-and-mlops"></a>

### Data Flywheel (The Most Important Concept)

```
Users interact with voice AI
        |
        v
Log audio + transcripts (with consent)
        |
        v
Identify failure cases (low confidence, user corrections, fallbacks)
        |
        v
Human review & annotation of hard cases
        |
        v
Add to training data + retrain
        |
        v
Deploy improved model --> Better user experience --> More users
```

### Monitoring Dashboard (Mention This!)

1. **Real-time:**
   - ASR confidence distribution
   - Latency percentiles per component
   - Error rates by language/accent
   - GPU utilization, memory usage

2. **Daily:**
   - WER on sampled traffic
   - Intent classification accuracy
   - User satisfaction trends
   - Data drift detection (audio characteristics changing)

3. **Weekly:**
   - Model performance by demographic segment
   - Cost per conversation
   - Capacity planning metrics

### Model Deployment Strategy

```
Research Model
    |
    v
[Offline Evaluation] -- WER on test sets, MOS evaluation
    |
    v
[Shadow Deployment] -- Run new model alongside production, compare outputs
    |
    v
[Canary Release] -- 1% -> 5% -> 25% -> 100% traffic
    |
    v
[Full Deployment] -- with automatic rollback on metric degradation
```

### Common Production Issues (Shows Experience!)

1. **Audio codec mismatch**: Training on 16kHz WAV, serving 8kHz telephony opus
2. **Background noise**: Office vs. car vs. street -- need robust preprocessing
3. **Accent/dialect drift**: User demographics change over time
4. **Adversarial inputs**: People testing the system with unusual speech
5. **Memory leaks**: Streaming ASR sessions that don't properly clean up
6. **Cold start**: GPU model loading takes 5-30 seconds (use model pre-warming)

---

## 9. Leadership & Tech Lead Angles <a name="9-leadership-and-tech-lead-angles"></a>

### As a Tech Lead, You Must Discuss:

#### Build vs. Buy Decisions
| Component | Build When... | Buy/Use When... |
|-----------|--------------|-----------------|
| ASR | Core differentiator, custom domain | General purpose, time-to-market matters |
| TTS | Unique voice is key product feature | Standard voices are acceptable |
| NLU | Proprietary intent taxonomy | Using LLM APIs (OpenAI, Anthropic) |
| Infrastructure | Scale requires custom optimization | Cloud providers suffice (AWS, GCP) |

**Specific recommendations to mention:**
- **ASR**: Start with Whisper/Deepgram API, build custom Conformer if ASR is core to product
- **TTS**: Start with ElevenLabs/Play.ht API, build custom if voice is core brand
- **LLM**: Use API (OpenAI/Anthropic) for NLU/Dialog, fine-tune open-source for cost/latency
- **Infrastructure**: NVIDIA Triton for model serving, Redis for caching, Kafka for streaming

#### Team Structure for Voice AI
```
Tech Lead (You)
    |
    +-- ASR Team (2-3 engineers)
    |       - Model training & optimization
    |       - Data pipeline & annotation
    |
    +-- TTS Team (1-2 engineers)
    |       - Voice quality & latency
    |       - Voice cloning/customization
    |
    +-- Dialog/NLU Team (2-3 engineers)
    |       - Intent understanding
    |       - Conversation management
    |       - LLM integration
    |
    +-- Platform/Infra Team (2-3 engineers)
            - Serving infrastructure
            - Monitoring & observability
            - Data pipeline
```

#### Technical Decision-Making Framework
1. **Reversibility**: Prefer reversible decisions (API -> custom model is easier than custom -> API)
2. **Time-to-value**: Ship MVP with APIs, iterate with custom models
3. **Data moat**: Own the data pipeline from day 1, even if using third-party models
4. **Modularity**: Design for component swapability (today's Whisper, tomorrow's custom model)

#### Communicating Technical Decisions
- "I would start with X because... and migrate to Y when we reach Z scale"
- "The trade-off here is latency vs. accuracy. For our use case, I'd prioritize..."
- "This is a two-way door decision, so I'd go fast and iterate"
- "We need to instrument this from day 1 to build our data flywheel"

---

## 10. Speed-Run Study Plan (2 Weeks) <a name="10-speed-run-study-plan"></a>

### Week 1: Foundations & Framework

| Day | Focus | Action |
|-----|-------|--------|
| **Day 1** | ML System Design Framework | Study the template in Section 3. Practice requirements gathering on 3 prompts |
| **Day 2** | ASR Deep Dive | Read Whisper paper, understand Conformer architecture, know CTC vs. RNN-T vs. attention |
| **Day 3** | TTS Deep Dive | Read VITS paper, understand mel spectrograms, know autoregressive vs. non-autoregressive trade-off |
| **Day 4** | Voice AI Stack | Map out the full pipeline (Section 2). Understand data flow, latency at each step |
| **Day 5** | Practice Design #1 | Design a voice assistant (Q1 from Section 5). Time yourself to 45 min |
| **Day 6** | Practice Design #2 | Design a call center voice AI (Q5). Focus on scale & production |
| **Day 7** | Review & Gaps | Identify weak areas, re-study |

### Week 2: Advanced & Polish

| Day | Focus | Action |
|-----|-------|--------|
| **Day 8** | LLM Integration | How LLMs fit into voice AI pipeline, function calling, RAG for knowledge |
| **Day 9** | Production & MLOps | Study Section 8. Know monitoring, deployment, data flywheel |
| **Day 10** | Latency Optimization | Master Section 6. Be able to give specific numbers |
| **Day 11** | Practice Design #3 | Design real-time translation (Q4). Include edge cases |
| **Day 12** | Practice Design #4 | Design voice cloning system (Q3). Discuss ethics |
| **Day 13** | Tech Lead Scenarios | Practice build-vs-buy discussions, team structuring, prioritization |
| **Day 14** | Mock Interview | Full 45-min mock with a friend or rubber duck. Record yourself |

### Ultra-Fast 3-Day Version (If Short on Time)

- **Day 1**: Memorize Sections 1-3 (Framework + Voice AI Stack + Template)
- **Day 2**: Practice 2 system designs with the template (Q1 and Q5)
- **Day 3**: Study Section 6 (Latency) and Section 9 (Tech Lead angles)

---

## 11. Resources & References <a name="11-resources-and-references"></a>

### Papers to Skim (Don't Deep Read, Know the Key Ideas)
1. **Whisper** -- "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI, 2022)
2. **Conformer** -- "Conformer: Convolution-augmented Transformer for Speech Recognition" (Google, 2020)
3. **VITS** -- "Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS" (2021)
4. **RNN-Transducer** -- "Sequence Transduction with Recurrent Neural Networks" (Graves, 2012)
5. **VALL-E** -- "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (Microsoft, 2023)
6. **SeamlessM4T** -- "Massively Multilingual & Multimodal Machine Translation" (Meta, 2023)

### Tools & Frameworks to Know
- **NVIDIA NeMo**: End-to-end toolkit for ASR, TTS, NLP
- **ESPnet**: Academic toolkit, great for understanding architectures
- **Coqui TTS** (archived but still useful): Open-source TTS
- **Hugging Face Transformers**: Whisper, wav2vec2, SpeechT5
- **NVIDIA Triton**: Model serving
- **Deepgram / AssemblyAI**: Commercial ASR APIs (know competitive landscape)
- **ElevenLabs / Play.ht**: Commercial TTS APIs

### Books & Courses
- **"Designing Machine Learning Systems"** by Chip Huyen (THE book for ML system design)
- **Stanford CS224S**: Spoken Language Processing
- **Coursera "ML System Design"** by educative.io

### Practice Platforms
- **Interviewing.io**: Mock interviews with real engineers
- **Pramp**: Free mock interviews
- **Record yourself**: Use voice memos, listen back, iterate

---

## Quick Reference Cheat Sheet

### When They Ask "Design X", Always Cover:

```
[x] Requirements: functional + non-functional + constraints
[x] Data: collection, annotation, augmentation, privacy
[x] Model: architecture choice with trade-offs
[x] Serving: latency, throughput, scaling strategy
[x] Evaluation: offline metrics, online metrics, A/B testing
[x] Monitoring: drift detection, failure modes, alerting
[x] Iteration: data flywheel, model improvement cycle
[x] Tech Lead: build vs buy, team structure, timeline
```

### Numbers to Have Ready

| Metric | Good Target | Impressive Target |
|--------|-------------|-------------------|
| ASR WER (general English) | < 10% | < 5% |
| ASR WER (medical/specialized) | < 15% | < 8% |
| TTS MOS | > 3.8 | > 4.2 |
| End-to-end latency | < 1500ms | < 800ms |
| TTS RTF | < 0.5 | < 0.1 |
| ASR RTF (streaming) | < 0.3 | < 0.1 |
| Concurrent sessions/GPU | 10-20 | 50+ |

### The One-Liner Trade-offs to Drop

- "Streaming ASR gives us lower latency but slightly higher WER compared to batch processing"
- "Non-autoregressive TTS is 10x faster but slightly less natural than autoregressive"
- "On-device inference eliminates network latency but limits model size to ~50-200MB"
- "Using an LLM for NLU is more flexible but adds 200-500ms; a fine-tuned BERT is 10x faster for known intents"
- "INT8 quantization gives us 2-4x speedup with < 1% accuracy degradation for most voice models"

---

**Final tip:** The interviewer for a Tech Lead role cares less about whether you know the exact architecture of Conformer, and more about whether you can **make sound technical decisions, articulate trade-offs, and lead a team to build and ship a reliable voice AI system.** Show judgment, not just knowledge.

Good luck -- you've got this!
