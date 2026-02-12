# ML System Design for Voice AI Engineer Tech Lead -- Complete Preparation Guide

> A no-fluff, deeply technical playbook for acing ML system design interviews focused on Voice AI / Speech / Audio ML.

---

## Table of Contents

1. [The Interview Format & What They Actually Evaluate](#1-the-interview-format)
2. [The Universal Framework (Your 45-Minute Skeleton)](#2-the-universal-framework)
3. [Voice AI Domain Deep-Dives](#3-voice-ai-domain-deep-dives)
4. [Core Voice AI System Design Problems (with Walkthroughs)](#4-core-voice-ai-system-design-problems)
5. [Key Architecture Patterns for Voice/Audio Systems](#5-key-architecture-patterns)
6. [The "Tech Lead" Multiplier -- What Sets You Apart](#6-the-tech-lead-multiplier)
7. [Metrics & Evaluation -- The Secret Weapon](#7-metrics--evaluation)
8. [Data Pipeline & Feature Engineering for Audio](#8-data-pipeline--feature-engineering)
9. [Production & MLOps for Voice Systems](#9-production--mlops)
10. [Study Plan (2-Week Sprint & 4-Week Deep)](#10-study-plan)
11. [Resources & References](#11-resources--references)
12. [Common Mistakes & How to Avoid Them](#12-common-mistakes)
13. [Cheat Sheets](#13-cheat-sheets)

---

## 1. The Interview Format

### What to Expect

| Aspect | Details |
|--------|---------|
| **Duration** | 45-60 minutes |
| **Format** | Open-ended problem -> you drive the conversation |
| **Whiteboard** | Virtual or physical diagram expected |
| **Depth vs Breadth** | Tech Lead = both. You must go deep on 2-3 components AND show full-system thinking |

### Evaluation Axes (ranked by weight for Tech Lead)

1. **System Architecture & Trade-offs** (30%) -- Can you design end-to-end?
2. **ML Depth** (25%) -- Do you know the models, losses, and why?
3. **Leadership Signals** (20%) -- Prioritization, phased rollout, team considerations
4. **Production Mindset** (15%) -- Latency, monitoring, failure modes
5. **Communication** (10%) -- Structured, clear, collaborative

---

## 2. The Universal Framework

Use this skeleton for EVERY ML system design question. Memorize it cold. Practice until it's muscle memory.

### Phase 1: Clarify & Scope (3-5 min)

```
CHECKLIST:
- What is the business goal / user problem?
- Who is the user? (end-user, internal, API consumer)
- What are the input/output modalities? (audio -> text, audio -> audio, text -> audio)
- What are the latency requirements? (real-time streaming vs batch)
- What scale? (QPS, concurrent users, hours of audio/day)
- What languages / accents / domains?
- What devices / environments? (mobile, call center, car, smart speaker)
- What is the noise profile? (clean studio, noisy street, far-field)
- Is there existing data? Labels? How much?
- On-device vs cloud vs hybrid?
```

**Pro Tip:** Frame 2-3 clarifying questions as "tradeoff-aware" questions:
> "Should we optimize for latency or accuracy? For a real-time voice assistant, I'd lean toward streaming with partial results, but if this is offline transcription, we can use full-context models."

### Phase 2: High-Level Architecture (5-7 min)

Draw the end-to-end system:

```
[Client/Device] --> [Audio Capture & Preprocessing]
                         |
                    [Streaming Gateway / Load Balancer]
                         |
                    [ML Inference Service(s)]
                         |
                    [Post-Processing / Business Logic]
                         |
                    [Response Delivery / TTS / Action]
                         |
                    [Logging / Monitoring / Feedback Loop]
```

**Key:** Name every box. For each box, say ONE sentence about what it does. Don't go deep yet.

### Phase 3: Data Pipeline & Features (5-7 min)

- Data collection strategy
- Labeling (human-in-the-loop, semi-supervised, active learning)
- Feature extraction (raw waveform vs spectral features)
- Data augmentation
- Train/val/test splits (careful with audio -- speaker-level splits!)

### Phase 4: Model Architecture & Training (10-12 min) -- GO DEEP HERE

- Model selection with justification
- Loss function choice
- Training strategy (pre-training, fine-tuning, multi-task)
- Key hyperparameters and how to tune
- Offline evaluation metrics

### Phase 5: Serving & Inference (5-7 min)

- Serving infrastructure (streaming vs batch)
- Model optimization (quantization, distillation, ONNX, TensorRT)
- Latency budget breakdown
- Scaling strategy
- Failover and graceful degradation

### Phase 6: Evaluation & Iteration (5-7 min)

- Offline metrics
- Online metrics (A/B testing)
- Monitoring & alerting
- Feedback loops
- Phased rollout plan

### Phase 7: Extensions & Trade-offs (remaining time)

- What would you do with 10x more data?
- How would you handle edge cases?
- Multi-language / multi-accent support
- Privacy & compliance (GDPR, data residency)
- On-device vs cloud trade-offs
- Team structure and work breakdown

---

## 3. Voice AI Domain Deep-Dives

### 3.1 Automatic Speech Recognition (ASR)

**Modern Architecture Landscape:**

| Generation | Architecture | Examples | Key Idea |
|-----------|-------------|----------|----------|
| Traditional | HMM-DNN hybrid | Kaldi | Acoustic model + Language model + Decoder (separate) |
| End-to-End (Seq2Seq) | Encoder-Decoder with Attention | LAS (Listen, Attend, Spell) | Single model, attention-based |
| End-to-End (CTC) | Encoder + CTC loss | DeepSpeech, Wav2Vec2 | No explicit alignment needed |
| End-to-End (Transducer) | RNN-T / Conformer-T | Google USM, Whisper-like | **Best for streaming** -- causal + label history |
| Foundation Models | Large pre-trained | Whisper, USM, SeamlessM4T | Massive pre-training, few-shot capable |

**What to know cold:**
- **CTC vs Attention vs Transducer** -- trade-offs for streaming
- **Conformer** architecture (Conv + Transformer = best of both worlds for audio)
- **Whisper** architecture and its strengths/weaknesses (not streaming-native!)
- **Wav2Vec 2.0 / HuBERT** -- self-supervised pre-training for speech
- Beam search decoding, language model fusion (shallow fusion, deep fusion)
- Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor (RTF)

**Streaming ASR specifics:**
- Chunk-based processing with look-ahead
- Endpoint detection (Voice Activity Detection + end-of-query detection)
- Partial/incremental results
- Latency: user-perceived latency = audio latency + network + compute + endpoint detection

### 3.2 Text-to-Speech (TTS)

**Modern Architecture Landscape:**

| Component | Options | State-of-Art |
|-----------|---------|--------------|
| **Text Analysis** | G2P, text normalization | Character-level or phoneme-level |
| **Acoustic Model** | Tacotron 2, FastSpeech 2, VITS, VALL-E | VITS (end-to-end), VALL-E (zero-shot) |
| **Vocoder** | WaveNet, WaveRNN, HiFi-GAN, Vocos | HiFi-GAN (fast + high quality) |
| **End-to-End** | VITS, NaturalSpeech 2/3, XTTS | Trending toward fully end-to-end |

**What to know cold:**
- **Two-stage vs End-to-End:** Acoustic model + Vocoder vs single model (VITS)
- **Mel spectrograms** as intermediate representation
- **Duration modeling** (why FastSpeech 2 is better for production than Tacotron 2 -- no attention alignment issues)
- **Voice cloning** approaches: speaker embeddings, few-shot (VALL-E), zero-shot
- **Evaluation:** MOS (Mean Opinion Score), PESQ, STOI, speaker similarity (cosine distance on embeddings)
- **Streaming TTS:** chunked synthesis, incremental vocoding

### 3.3 Speaker Verification / Identification

**Architecture:**

```
Audio --> Feature Extraction --> Speaker Embedding Network --> Embedding Vector
                                                                    |
                                                        [Cosine Similarity / PLDA]
                                                                    |
                                                          [Accept / Reject]
```

**Key concepts:**
- **Speaker embeddings:** d-vectors (DNN), x-vectors (TDNN), ECAPA-TDNN (state-of-art)
- **Training losses:** Softmax, AM-Softmax, AAM-Softmax (additive angular margin)
- **Verification vs Identification:** 1:1 matching vs 1:N search
- **Enrollment:** few-shot speaker profile creation
- **Metrics:** EER (Equal Error Rate), minDCF, TAR@FAR
- **Spoofing detection:** anti-spoofing as a critical companion system

### 3.4 Voice Activity Detection (VAD)

- Seemingly simple but CRITICAL for pipeline performance
- **Models:** WebRTC VAD, Silero VAD, custom RNN/CNN models
- **Frame-level** binary classification with smoothing
- **Hangover schemes** to avoid cutting off speech
- Energy-based vs model-based

### 3.5 Audio/Speech Enhancement & Separation

- **Noise suppression:** spectral gating, deep learning (DCCRN, FullSubNet, DTLN)
- **Echo cancellation:** adaptive filters + neural post-processing
- **Source separation:** Conv-TasNet, SepFormer
- **Dereverberation:** WPE, neural approaches
- **Beamforming:** MVDR, neural beamforming for multi-mic

### 3.6 Spoken Language Understanding (SLU)

- **Pipeline:** ASR -> NLU (intent + entities)
- **End-to-End SLU:** directly from audio to intent (skipping ASR)
- **Trade-off:** Pipeline is more debuggable; E2E can recover from ASR errors
- **Dialogue management:** state tracking, policy learning

### 3.7 Voice Conversion & Audio Deepfakes

- **Voice conversion:** change speaker identity while preserving content
- **Detection:** audio deepfake / spoofing detection systems
- Growing importance for security and trust

---

## 4. Core Voice AI System Design Problems

### Problem 1: "Design a Real-Time Voice Assistant" (Most Common!)

**Scope:** Think Alexa, Google Assistant, Siri -- end-to-end.

```
                    +-------------------+
                    |   Wake Word       |
                    |   Detection       | (always-on, on-device)
                    +--------+----------+
                             |
                    +--------v----------+
                    |   Audio Streaming |
                    |   to Cloud        | (WebSocket / gRPC streaming)
                    +--------+----------+
                             |
              +--------------v--------------+
              |                             |
     +--------v--------+          +--------v--------+
     | Voice Activity   |          | Audio           |
     | Detection (VAD)  |          | Enhancement     |
     +--------+---------+          +--------+--------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+
                   |   Streaming     |
                   |   ASR (RNN-T)   |
                   +--------+--------+
                            |
                   +--------v--------+
                   |   NLU / Intent  |
                   |   Classification|
                   +--------+--------+
                            |
                   +--------v--------+
                   |   Dialog Manager|
                   |   / Orchestrator|
                   +--------+--------+
                            |
              +-------------+-------------+
              |             |             |
        +-----v-----+ +----v----+ +------v------+
        | API/Action | | Knowledge| |   Response  |
        | Execution  | | Graph    | |  Generation |
        +-----+------+ +----+----+ +------+------+
              |             |             |
              +-------------+-------------+
                            |
                   +--------v--------+
                   |   TTS / Neural  |
                   |   Speech Synth  |
                   +--------+--------+
                            |
                   +--------v--------+
                   |   Audio Stream  |
                   |   to Device     |
                   +-----------------+
```

**Key design decisions to discuss:**
1. **Wake word on-device:** Small keyword spotting model (< 1MB), always-on, low power
2. **Streaming protocol:** gRPC bidirectional streaming preferred (lower latency than REST)
3. **ASR choice:** RNN-Transducer with Conformer encoder for streaming
4. **Latency budget:**
   - Wake word detection: < 200ms
   - Audio to cloud: ~50ms (depends on network)
   - ASR first partial: < 300ms from speech start
   - End-to-end (speech end to response start): < 1.5s target
5. **Endpoint detection:** Combined VAD + silence duration + semantic completeness
6. **TTS streaming:** Start speaking before full response is generated (chunked)

### Problem 2: "Design a Speaker Verification System for Banking"

**Critical requirements:** Security, low false accept rate, spoofing resistance

```
ENROLLMENT FLOW:
[User speaks passphrase x3] -> [VAD] -> [Feature Extract] -> [Speaker Embedding Model]
                                                                      |
                                                          [Average embeddings]
                                                                      |
                                                          [Store encrypted voiceprint]

VERIFICATION FLOW:
[User speaks] -> [VAD] -> [Anti-Spoofing Check] -> [Feature Extract] -> [Speaker Embedding]
                                    |                                          |
                              [Reject if spoof]                    [Compare to enrolled voiceprint]
                                                                          |
                                                                  [Score > threshold?]
                                                                    /           \
                                                              [Accept]      [Reject + Fallback]
```

**Key points:**
- **Operating point:** Set threshold for very low FAR (0.1%) even if FRR is higher (5%)
- **Anti-spoofing:** MANDATORY. Use AASIST or similar for replay/synthesis/conversion attacks
- **Channel robustness:** Enroll and verify over same channel (phone narrowband vs wideband)
- **Text-dependent vs text-independent:** Banks often use text-dependent (same passphrase)
- **Calibration:** Score normalization (Z-norm, T-norm) across populations
- **Privacy:** Store embeddings, never raw audio. Encryption at rest.
- **Fallback:** Multi-factor auth when voice fails

### Problem 3: "Design a Real-Time Speech Translation System"

```
[Source Speech] -> [Streaming ASR] -> [MT / LLM Translation] -> [Target TTS]
                        |                                            |
                   [Partial results]                         [Streaming audio out]

OR (End-to-End):

[Source Speech] -> [Speech-to-Speech Translation Model] -> [Target Speech]
                   (e.g., SeamlessM4T, Translatotron 2)
```

**Key trade-offs:**
- **Cascade vs E2E:** Cascade is more flexible/debuggable; E2E preserves prosody/emotion
- **Latency:** Cascade adds up; E2E can be more parallel
- **Streaming considerations:** Re-translation strategy (as more context arrives, revise translation)
- **Voice preservation:** Keep source speaker's voice characteristics in target language

### Problem 4: "Design a Call Center Voice Analytics Platform"

```
[Call Recording] --> [Diarization] --> [ASR per speaker] --> [NLP Analytics]
       |                                                          |
  [Audio Quality     [Speaker         [Transcription]     [Sentiment Analysis]
   Check]            Clustering]                          [Topic Detection]
                                                          [Compliance Check]
                                                          [Agent Scoring]
                                                          [Call Summarization]
```

**Batch vs real-time considerations:**
- Post-call analytics: batch, optimize for accuracy
- Real-time agent assist: streaming, optimize for latency
- **Speaker diarization:** Who spoke when? (Spectral clustering, EEND, TS-VAD)
- **Scale:** Thousands of hours per day, cost optimization matters

### Problem 5: "Design a Voice Cloning / Custom TTS System"

**Two approaches:**
1. **Fine-tuning:** Take base TTS, fine-tune on target speaker (needs 30min-1hr data)
2. **Zero-shot:** VALL-E / XTTS style -- provide 3-10 seconds of reference audio

**Key discussion points:**
- Data collection pipeline for voice talent
- Quality assurance (pronunciation, prosody, naturalness)
- Ethical considerations and consent
- Watermarking synthesized speech
- Adapting to new speakers without retraining the full model

---

## 5. Key Architecture Patterns

### Pattern 1: Streaming Audio Pipeline

```python
# Pseudocode for streaming audio processing
class StreamingASRPipeline:
    def __init__(self):
        self.vad = StreamingVAD(frame_duration_ms=30)
        self.enhancer = NoiseSuppressionModel()
        self.asr = StreamingTransducer(
            encoder=ConformerEncoder(chunk_size=640ms, left_context=6s),
            decoder=TransducerDecoder(beam_size=4)
        )
        self.endpointer = Endpointer(
            silence_threshold_ms=700,
            max_utterance_ms=30000
        )

    def process_chunk(self, audio_chunk):
        # Step 1: VAD -- is anyone speaking?
        is_speech = self.vad(audio_chunk)

        if not is_speech and not self.in_utterance:
            return None  # Nothing to do

        # Step 2: Enhancement
        clean_audio = self.enhancer(audio_chunk)

        # Step 3: ASR -- get partial transcript
        partial_result = self.asr.process_chunk(clean_audio)

        # Step 4: Endpointing -- is the utterance complete?
        is_final = self.endpointer.check(is_speech, partial_result)

        if is_final:
            final_result = self.asr.finalize()
            return StreamingResult(text=final_result, is_final=True)

        return StreamingResult(text=partial_result, is_final=False)
```

### Pattern 2: Latency Budget Breakdown

Always present this in your design. It shows production maturity.

```
Component               | Target | Notes
------------------------|--------|------
Audio capture + encode  |  20ms  | Opus codec, 20ms frames
Network (up)            |  50ms  | Depends on connectivity
VAD + Enhancement       |  10ms  | Lightweight models
ASR first partial       | 200ms  | Conformer-T, chunk-based
ASR final (after EPD)   | 300ms  | Includes endpoint detection
NLU / Intent            |  50ms  | BERT-small or distilled
Dialog + API call       | 200ms  | Depends on backend
TTS synthesis           | 150ms  | Time-to-first-audio
Network (down)          |  50ms  |
Audio playback start    |  20ms  |
------------------------|--------|------
TOTAL (speech end       | ~750ms | Target < 1 second
 to response start)     |        |
```

### Pattern 3: Model Serving for Audio

```
                    +------------------+
                    |  Load Balancer   |
                    |  (L7, gRPC-aware)|
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v-------+
     | GPU Worker  |  | GPU Worker  |  | GPU Worker   |
     | (ASR model) |  | (ASR model) |  | (ASR model)  |
     | Triton /    |  | Triton /    |  | Triton /     |
     | TorchServe  |  | TorchServe  |  | TorchServe   |
     +-------------+  +-------------+  +--------------+
```

**Serving considerations:**
- **Dynamic batching:** Group audio frames from multiple streams for GPU efficiency
- **Model format:** ONNX Runtime or TensorRT for inference optimization
- **Quantization:** INT8 for encoder, FP16 for decoder typically
- **GPU sharing:** Multiple model instances per GPU with careful memory management
- **CPU fallback:** Some components (VAD, feature extraction) better on CPU

### Pattern 4: Feedback Loop & Active Learning

```
[Production Traffic] --> [Log Audio + Predictions]
         |
    [Sample Selection]  <-- Active Learning Strategy
         |                   (uncertainty, diversity, error-prone)
         |
    [Human Review / Transcription]
         |
    [Add to Training Data]
         |
    [Retrain / Fine-tune]
         |
    [Shadow Evaluation] --> [A/B Test] --> [Full Rollout]
```

---

## 6. The "Tech Lead" Multiplier

This is what separates a senior IC answer from a Tech Lead answer:

### 6.1 Prioritization & Phased Approach

ALWAYS present your design in phases:

**Phase 1 (MVP - 4 weeks):**
- Use off-the-shelf ASR (Whisper API / Google STT)
- Simple rule-based NLU
- Basic TTS (gTTS or pre-built)
- Goal: validate product-market fit

**Phase 2 (V1 - 3 months):**
- Fine-tune open-source ASR on domain data
- Train custom NLU model
- Deploy on-prem for latency/privacy
- Add monitoring and logging

**Phase 3 (Scale - 6 months):**
- Custom streaming ASR model
- Neural TTS with brand voice
- Active learning pipeline
- Multi-language support
- On-device components for latency-critical paths

### 6.2 Team Structure Discussion

Volunteer this even if not asked:

> "For this system, I'd structure the team as:
> - 2 ML Engineers focused on ASR model training and optimization
> - 1 ML Engineer on TTS / audio synthesis
> - 1 ML Engineer on speaker/audio understanding models
> - 2 Backend Engineers on serving infrastructure and streaming
> - 1 Data Engineer on pipeline and labeling infrastructure
> - Shared: 0.5 PM, 0.5 Data Annotation Lead
>
> As tech lead, I'd own the architecture decisions, model selection, and coordinate cross-cutting concerns like the streaming protocol and latency budget."

### 6.3 Technical Risk Assessment

Identify and address risks:

| Risk | Mitigation |
|------|-----------|
| ASR accuracy on domain-specific terms | Custom LM, contextual biasing, user corrections |
| Latency spikes under load | Auto-scaling, model optimization, graceful degradation |
| Accent/dialect coverage gaps | Targeted data collection, accent-specific fine-tuning |
| Model regression on update | Shadow evaluation, canary deployment, automatic rollback |
| Privacy/compliance | On-device processing where possible, data retention policies |

### 6.4 Make vs Buy Decisions

Show you think about ROI:

| Component | Build | Buy | Recommendation |
|-----------|-------|-----|----------------|
| ASR | Full control, lower per-unit cost at scale | Faster, proven quality | Buy for MVP, build for scale |
| TTS | Custom voice, full control | Fast, good quality | Buy unless custom voice is core differentiator |
| VAD | Simple, well-solved | Very simple to integrate | Buy (Silero VAD is free & great) |
| NLU | Domain-specific needs | Generic, may not fit | Build (domain knowledge is competitive advantage) |
| Infra | Custom optimization | Battle-tested | Buy infra (Triton, K8s), build orchestration |

---

## 7. Metrics & Evaluation

### 7.1 Offline Metrics

| Task | Metric | What It Measures | Target Range |
|------|--------|-----------------|-------------|
| ASR | WER (Word Error Rate) | % words wrong | 5-15% (domain dependent) |
| ASR | CER (Character Error Rate) | % characters wrong | Often lower than WER |
| ASR | RTF (Real-Time Factor) | Processing time / audio duration | < 0.3 for real-time |
| TTS | MOS (Mean Opinion Score) | Human quality rating (1-5) | > 4.0 for production |
| TTS | PESQ / POLQA | Objective speech quality | Correlates with MOS |
| Speaker Verification | EER (Equal Error Rate) | Error rate at FAR=FRR | < 3% for good systems |
| Speaker Verification | minDCF | Weighted decision cost | Lower is better |
| VAD | Frame-level F1 | Speech detection accuracy | > 0.95 |
| Diarization | DER (Diarization Error Rate) | Misattribution + missed + false alarm | < 10% |
| NLU | Intent Accuracy | Correct intent classification | > 95% |
| NLU | Entity F1 | Named entity extraction | > 90% |

### 7.2 Online Metrics (Production)

- **Task completion rate:** Did the user achieve their goal?
- **Turn count:** Fewer turns = better understanding
- **Latency percentiles:** p50, p90, p99
- **Barge-in rate:** User interrupting the system (signals frustration)
- **Repeat rate:** User repeating themselves (signals ASR failure)
- **Fallback rate:** How often system says "I didn't understand"
- **User satisfaction:** Explicit ratings or implicit signals
- **Retention / engagement:** Long-term success metric

### 7.3 How to Present Metrics in Interview

> "I'd set up a metrics hierarchy:
> 1. **North Star:** Task completion rate (business impact)
> 2. **Model metrics:** WER for ASR, MOS for TTS (ML quality)
> 3. **System metrics:** p99 latency, availability (operational health)
> 4. **Guardrail metrics:** Toxicity rate, PII leak rate (safety)
>
> We'd monitor all in real-time, alert on model metrics, and report north star weekly."

---

## 8. Data Pipeline & Feature Engineering

### 8.1 Audio Feature Extraction

```
Raw Waveform (16kHz, 16-bit)
    |
    v
Pre-emphasis (optional, high-pass filter)
    |
    v
Framing (25ms window, 10ms hop)
    |
    v
Windowing (Hamming / Hanning)
    |
    v
FFT (512 or 1024 point)
    |
    v
+------------------+------------------+
|                  |                  |
v                  v                  v
Mel Filterbank     MFCC              Log Power
(80 bins)          (13-40 coeffs)    Spectrogram
|                  |
v                  v
Log Mel            + Delta +
Spectrogram        Delta-Delta
```

**Modern trend:** Many SOTA models (Wav2Vec2, Whisper, etc.) take raw waveforms or simple log-mel features and learn their own representations. But knowing the traditional pipeline shows depth.

### 8.2 Data Augmentation for Audio (Critical!)

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **SpecAugment** | Mask time/frequency bands in spectrogram | Regularization, robustness |
| **Speed perturbation** | Resample at 0.9x, 1.0x, 1.1x | Speaker variation |
| **Noise addition** | Mix with MUSAN/AudioSet noise | Noise robustness |
| **Room simulation** | Convolve with RIR (Room Impulse Response) | Reverberation robustness |
| **Codec augmentation** | Encode/decode with various codecs | Channel robustness |
| **Pitch shifting** | Change pitch without changing speed | Speaker variation |
| **Volume perturbation** | Random gain adjustment | Level robustness |
| **Time stretching** | Change speed without changing pitch | Duration variation |

### 8.3 Data Collection Strategy

**For a new voice AI product:**

1. **Bootstrap:** Use public datasets (LibriSpeech, Common Voice, VoxPopuli)
2. **Domain data:** Record/collect domain-specific audio (medical terms, financial jargon)
3. **Synthetic data:** TTS-generated speech for rare entities/phrases
4. **Production data:** Log (with consent) and selectively label production audio
5. **Active learning:** Prioritize labeling samples where the model is uncertain

**Speaker-level train/test split:**
> "This is a common gotcha -- you MUST split by speaker, not by utterance. Otherwise, the model memorizes speaker characteristics and inflates metrics."

---

## 9. Production & MLOps

### 9.1 Serving Architecture Choices

| Approach | Latency | Cost | Complexity | Use Case |
|----------|---------|------|-----------|----------|
| **NVIDIA Triton** | Very low | GPU cost | Medium | High-throughput, multi-model |
| **TorchServe** | Low | GPU cost | Low | PyTorch models, simpler setup |
| **ONNX Runtime** | Low | CPU/GPU | Low | Cross-framework, edge deployment |
| **TensorRT** | Lowest | GPU cost | High | Maximum throughput needed |
| **OpenVINO** | Low | CPU optimized | Medium | Intel hardware, edge |
| **Custom gRPC** | Flexible | Depends | High | Full control needed |

### 9.2 Model Optimization Checklist

```
1. [ ] Export to ONNX or TorchScript
2. [ ] Apply FP16 mixed precision (almost always a win)
3. [ ] Try INT8 quantization (test quality carefully!)
4. [ ] Knowledge distillation for smaller model
5. [ ] Prune attention heads / layers if applicable
6. [ ] Optimize for target hardware (TensorRT for NVIDIA, CoreML for Apple)
7. [ ] Profile and eliminate bottlenecks
8. [ ] Enable dynamic batching in serving framework
9. [ ] Test latency under realistic load (not just single-request)
10. [ ] Set up model warmup to avoid cold start latency
```

### 9.3 Monitoring Dashboard

```
REAL-TIME MONITORING:
+--------------------------------------------------+
|  Request Rate: 1,247 req/s    Error Rate: 0.03%  |
|  p50 Latency:  124ms          p99 Latency: 387ms |
|  GPU Util:     73%            Memory:      81%   |
+--------------------------------------------------+
|  ASR Confidence Distribution  |  WER (sampled)   |
|  [histogram]                  |  Today: 8.2%     |
|                               |  7-day: 8.1%     |
+--------------------------------------------------+
|  Alerts:                                          |
|  - WER spike > 2 std dev from baseline            |
|  - p99 latency > 500ms for 5 min                 |
|  - Confidence < 0.5 rate > 10%                    |
|  - GPU memory > 90%                               |
+--------------------------------------------------+
```

### 9.4 A/B Testing for Voice Systems

**Challenges unique to voice:**
- Can't easily show two versions side-by-side (unlike visual UI)
- Quality is subjective -- need sufficient sample size
- Latency differences are felt immediately
- Must test on diverse speaker populations

**Approach:**
1. User-level randomization (not request-level -- consistency matters)
2. Stratify by: device type, language, accent, usage pattern
3. Run for at least 1-2 weeks (capture weekly patterns)
4. Primary metric: task completion rate
5. Guardrail: latency p99 must not regress > 10%

---

## 10. Study Plan

### 2-Week Sprint (Minimum Viable Preparation)

**Week 1: Foundations**

| Day | Focus | Activity |
|-----|-------|---------|
| 1 | Framework | Memorize and practice the universal framework (Section 2). Do 2 timed practice problems. |
| 2 | ASR Deep Dive | Read about CTC, Attention, Transducer. Understand Conformer architecture. Draw the diagrams. |
| 3 | TTS Deep Dive | Understand Tacotron 2 -> FastSpeech 2 -> VITS evolution. Know mel spectrograms. |
| 4 | Speaker & Audio | Speaker verification (x-vectors, ECAPA-TDNN), VAD, noise suppression basics. |
| 5 | Practice Problem | "Design a Voice Assistant" -- full 45-min timed session. Record yourself. |
| 6 | Serving & Infra | Streaming architectures, gRPC, Triton, model optimization. Latency budgets. |
| 7 | Review | Re-do weak areas. Read one Whisper or Conformer paper in detail. |

**Week 2: Practice & Polish**

| Day | Focus | Activity |
|-----|-------|---------|
| 8 | Practice Problem | "Design a Call Center Analytics Platform" -- timed |
| 9 | Practice Problem | "Design a Speaker Verification System" -- timed |
| 10 | Metrics & Data | Master all metrics (Section 7). Practice explaining data pipelines. |
| 11 | Tech Lead Skills | Practice phased approaches, team structure, risk assessment. |
| 12 | Practice Problem | "Design a Real-Time Translation System" -- timed |
| 13 | Mock Interview | Find a friend or use a recording. Full 45-min session. |
| 14 | Final Review | Review cheat sheets. Rest. Trust your preparation. |

### 4-Week Deep Preparation

Add to the above:

**Week 3: Advanced Topics**
- Self-supervised learning for speech (Wav2Vec 2.0, HuBERT, data2vec)
- Multi-lingual and code-switching ASR
- On-device ML and model compression techniques
- Voice cloning and ethical AI
- Read 3-5 key papers in detail

**Week 4: Integration & Mock Interviews**
- 3-4 full mock interviews with different problems
- Practice explaining your past projects as system designs
- Prepare for follow-up deep dives on any component
- Study the specific company's voice AI products and technical blog posts

### Top Resources (Ranked by ROI)

| Resource | Type | Why |
|----------|------|-----|
| **"Designing Machine Learning Systems" by Chip Huyen** | Book | Best overall ML system design book |
| **Stanford CS224S (Spoken Language Processing)** | Course | Academic depth on speech |
| **Whisper paper (Radford et al., 2022)** | Paper | Modern ASR architecture + training at scale |
| **Conformer paper (Gulati et al., 2020)** | Paper | THE architecture for speech |
| **Machine Learning System Design Interview by Alex Xu** | Book | Great practice problems and framework |
| **NVIDIA NeMo documentation** | Docs | Production speech AI toolkit |
| **HuggingFace Audio Course** | Course | Hands-on, modern |
| **ESPnet / WeNet repos** | Code | See how production ASR is built |
| **Meta's SeamlessM4T paper** | Paper | Multi-modal, multi-lingual speech |
| **SpeechBrain documentation** | Docs | Great for speaker verification |

---

## 11. Resources & References

### Must-Read Papers

1. **ASR:**
   - "Attention Is All You Need" (Vaswani et al., 2017) -- foundation
   - "Conformer" (Gulati et al., 2020) -- speech-specific Transformer
   - "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, Radford et al., 2022)
   - "RNN-Transducer" (Graves, 2012) -- streaming ASR foundation
   - "wav2vec 2.0" (Baevski et al., 2020) -- self-supervised speech

2. **TTS:**
   - "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Tacotron 2, 2018)
   - "FastSpeech 2" (Ren et al., 2020) -- non-autoregressive TTS
   - "Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS" (VITS, Kim et al., 2021)
   - "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, Wang et al., 2023)
   - "HiFi-GAN" (Kong et al., 2020) -- vocoder

3. **Speaker:**
   - "ECAPA-TDNN" (Desplanques et al., 2020)
   - "In defence of metric learning for speaker recognition" (Chung et al., 2020)

4. **System Design:**
   - "Scaling Speech Technology to 1,000+ Languages" (Meta MMS, 2023)
   - Google USM (Universal Speech Model, 2023)

### Useful Blogs & Posts

- Google AI Blog -- speech and audio posts
- Meta AI Blog -- speech and audio research
- NVIDIA Developer Blog -- NeMo toolkit, optimization
- AssemblyAI Blog -- practical ASR engineering
- Deepgram Blog -- production speech AI

---

## 12. Common Mistakes

### Mistake 1: Jumping to Model Architecture Too Fast

**Wrong:** "I'd use Whisper for ASR..."

**Right:** "Let me first clarify the requirements. For a real-time assistant, we need streaming capability, so Whisper's encoder-decoder architecture isn't ideal -- it processes full utterances. I'd instead use a Conformer-Transducer which supports streaming naturally."

### Mistake 2: Ignoring Audio-Specific Challenges

Don't treat it like a generic ML system. Call out:
- Noise and channel variability
- Speaker diversity (accents, ages, speech impediments)
- Real-time constraints unique to audio
- The importance of VAD and endpointing

### Mistake 3: Not Discussing Streaming vs Batch

For voice AI, this is THE key architectural decision. Always address:
- Can the model process incrementally?
- What's the chunk size?
- How does look-ahead/left context affect quality vs latency?

### Mistake 4: Forgetting the Human Element

- Data labeling quality and cost
- Edge cases that need human fallback
- User experience of errors (ASR mistakes feel worse than visual typos)

### Mistake 5: Overcomplicating the Initial Design

Start simple, then add complexity. Show you can prioritize.

### Mistake 6: Not Drawing Diagrams

Always draw. Even in a phone screen, describe what you would draw. In virtual interviews, use the drawing tool extensively.

### Mistake 7: Ignoring Privacy & Ethics

Voice data is biometric data. Always mention:
- GDPR / CCPA implications
- Data retention policies
- User consent mechanisms
- Bias in speech recognition (accent, gender, age)
- Deepfake prevention for voice cloning systems

---

## 13. Cheat Sheets

### Quick Reference: Audio Basics

```
Sample Rate:     8kHz (telephone), 16kHz (standard speech), 44.1kHz (music), 48kHz (broadcast)
Bit Depth:       16-bit standard
Channels:        Mono for speech processing (usually)
Frame Duration:  25ms window, 10ms hop (standard for speech)
Mel Filterbanks: 80 bins (common for modern models)
FFT Size:        512 (for 16kHz) or 1024
Nyquist:         Sample rate / 2 = max representable frequency
Speech Range:    ~85Hz - 8kHz (fundamental + harmonics)
```

### Quick Reference: Model Sizes

| Model | Parameters | RTF (GPU) | Use Case |
|-------|-----------|-----------|----------|
| Silero VAD | ~1M | 0.001 | Voice activity detection |
| Whisper tiny | 39M | ~0.05 | Light ASR |
| Whisper base | 74M | ~0.1 | Balanced ASR |
| Whisper small | 244M | ~0.2 | Good quality ASR |
| Whisper medium | 769M | ~0.4 | High quality ASR |
| Whisper large-v3 | 1.55B | ~0.8 | Best quality ASR |
| Conformer-T (prod) | 100-300M | ~0.1-0.3 | Streaming ASR |
| ECAPA-TDNN | ~15M | ~0.01 | Speaker verification |
| FastSpeech 2 | ~30M | ~0.05 | Fast TTS |
| VITS | ~80M | ~0.1 | E2E TTS |
| HiFi-GAN | ~14M | ~0.01 | Vocoder |

### Quick Reference: Loss Functions

| Task | Loss | Notes |
|------|------|-------|
| ASR (CTC) | CTC Loss | Handles alignment; allows blank tokens |
| ASR (Seq2Seq) | Cross-Entropy + Attention | Teacher-forced decoding |
| ASR (Transducer) | Transducer/RNN-T Loss | Joint CTC-like + label history |
| TTS (Acoustic) | L1/L2 Mel Loss + Duration Loss | Spectrogram prediction |
| TTS (GAN) | L1 + Adversarial + Feature Matching | For vocoder training |
| TTS (VITS) | VAE (ELBO) + GAN + Duration | End-to-end |
| Speaker Verif | AAM-Softmax | Additive Angular Margin |
| VAD | Binary Cross-Entropy | Frame-level |
| Denoising | SI-SNR or L1 on waveform | Scale-invariant signal-to-noise |

### Quick Reference: Streaming vs Non-Streaming

| Aspect | Streaming | Non-Streaming |
|--------|-----------|--------------|
| Architecture | RNN-T, Streaming Conformer | Whisper, LAS |
| Latency | Low (< 500ms) | High (full utterance) |
| Accuracy | Slightly lower (limited context) | Higher (full context) |
| Complexity | Higher | Lower |
| Use Case | Voice assistant, live captions | Transcription, subtitles |
| Decoding | Greedy / limited beam | Full beam search + LM rescore |

### The "Golden Answer" Template

When you get a question, structure your answer like this:

```
1. CLARIFY (2-3 smart questions)
   "Before I design this, let me ask..."

2. DEFINE SUCCESS
   "The north star metric would be X, and we'd measure Y and Z"

3. HIGH-LEVEL ARCHITECTURE
   [Draw diagram]
   "At a high level, the system has these components..."

4. DEEP DIVE (pick the most interesting/challenging component)
   "Let me go deep on the [ASR/TTS/serving] component..."
   - Model choice + justification
   - Training data + augmentation
   - Key hyperparameters
   - Evaluation approach

5. PRODUCTION CONSIDERATIONS
   "For serving this at scale..."
   - Latency budget
   - Scaling strategy
   - Monitoring

6. PHASED APPROACH
   "I'd build this in phases..."
   Phase 1: MVP
   Phase 2: Quality
   Phase 3: Scale

7. TRADE-OFFS & EXTENSIONS
   "Some key trade-offs to consider..."
```

---

## Final Tips

1. **Practice out loud.** ML system design is a verbal exam. Writing notes is not enough.

2. **Time yourself.** 45 minutes goes fast. Spend no more than 5 minutes on clarification.

3. **Have 3-4 "go-to" designs memorized.** Voice assistant, speaker verification, call analytics, and speech translation cover 80% of questions.

4. **Know your "T-shape."** Be broad across all of voice AI, but pick one area where you can go arbitrarily deep (e.g., streaming ASR architecture).

5. **Prepare your "story."** As a tech lead, you'll be asked about past decisions. Have 2-3 stories ready about system design decisions you made, trade-offs you navigated, and how you led the technical direction.

6. **Study the company.** Know their voice AI products, their tech stack, their published papers. Reference them naturally in your design.

7. **Be opinionated but flexible.** "I'd recommend X because of Y, but if Z changes, we should consider W." This shows both conviction and adaptability.

8. **Don't memorize -- understand.** If you understand WHY Conformer works well for speech (convolution captures local patterns, attention captures global dependencies), you can reason about any new architecture.

9. **The interviewer is your collaborator, not your judge.** Engage them: "Does this seem like the right direction?" "Should I go deeper on X or move to Y?"

10. **Confidence comes from preparation.** Do the work. Trust the process. You've got this.

---

*Good luck with your Voice AI Tech Lead interview!*
