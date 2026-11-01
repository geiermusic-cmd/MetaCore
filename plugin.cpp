#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <atomic>
#include <vector>
#include <array>
#include <cmath>
#include <random>

static constexpr int kCanvasWidth  = 256;
static constexpr int kCanvasHeight = 128;
static constexpr int kFFTOrder     = 10; // 1024-point FFT
static constexpr int kFFTSize      = 1 << kFFTOrder;
static constexpr int kHopSize      = kFFTSize / 4; // 4x overlap

struct PixelRGB
{
    float r = 0.0f; // Left channel amplitude
    float g = 0.0f; // Right channel amplitude
    float b = 0.0f; // Modulation depth (FM index / Harmonic morph / Detune)
};

//==============================================================================
// PHASE 1: MULTI-ENGINE WAVETABLE, FM & SAMPLE/GRAIN VOICE BANK
//==============================================================================
class MetaVoiceBank
{
public:
    static constexpr int kNumRows   = 128;
    static constexpr int kTableSize = 4096;

    enum SynthType
    {
        Sine = 0,
        RichSaw,
        HollowSquare,
        FormantChoir,
        MetallicFM,
        FilteredNoise,
        SampleGrain
    };

    MetaVoiceBank()
    {
        buildWavetables();
        resetPhases();
    }

    void prepare(double sampleRate)
    {
        sr = sampleRate > 0.0 ? sampleRate : 44100.0;
        resetPhases();
    }

    void resetPhases()
    {
        std::mt19937 rng(1337);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int y = 0; y < kNumRows; ++y)
        {
            carrierPhase[y] = dist(rng);
            modPhase[y]     = dist(rng);
            grainPos[y]     = dist(rng) * 4096.0;
            noiseState1[y]  = 0.0f;
            noiseState2[y]  = 0.0f;
        }
    }

    void loadSampleBuffer(const juce::AudioBuffer<float>& newSample, double sourceSampleRate)
    {
        const juce::ScopedLock sl(sampleLock);
        if (newSample.getNumSamples() > 0)
        {
            sampleData.makeCopyOf(newSample);
            sampleSR = sourceSampleRate > 0.0 ? sourceSampleRate : 44100.0;
            hasSample = true;
        }
    }

    void renderSample(int synthType,
                      float modDepthKnob,
                      const double* rowFreqs,
                      const float* smoothL,
                      const float* smoothR,
                      const float* smoothB,
                      float& outL,
                      float& outR)
    {
        float sumL = 0.0f;
        float sumR = 0.0f;
        const double invSR = 1.0 / sr;

        for (int y = 0; y < kNumRows; ++y)
        {
            const float ampL = smoothL[y];
            const float ampR = smoothR[y];
            const double freq = rowFreqs[y];
            const double phaseInc = freq * invSR;

            double cPhase = carrierPhase[y] + phaseInc;
            cPhase -= std::floor(cPhase);
            carrierPhase[y] = cPhase;

            if (ampL <= 0.0001f && ampR <= 0.0001f)
                continue;

            const float modAmt = juce::jlimit(0.0f, 1.0f, smoothB[y] + modDepthKnob * 0.5f);
            float sig = 0.0f;

            switch (synthType)
            {
                case Sine:
                {
                    float fundamental = lookupTable(sineTable, cPhase);
                    if (modAmt > 0.01f)
                    {
                        double p2 = cPhase * 2.0;
                        p2 -= std::floor(p2);
                        sig = fundamental + modAmt * 0.45f * lookupTable(sineTable, p2);
                    }
                    else
                    {
                        sig = fundamental;
                    }
                    break;
                }

                case RichSaw:
                {
                    float s1 = lookupTable(sawTable, cPhase);
                    double mPhase = modPhase[y] + phaseInc * (1.0 + 0.006 * (1.0 + modAmt));
                    mPhase -= std::floor(mPhase);
                    modPhase[y] = mPhase;
                    float s2 = lookupTable(sawTable, mPhase);
                    sig = (s1 + s2 * (0.4f + 0.6f * modAmt)) * 0.65f;
                    break;
                }

                case HollowSquare:
                {
                    float sq = lookupTable(squareTable, cPhase);
                    double subP = cPhase * 0.5;
                    sig = sq * (1.0f - 0.3f * modAmt) + lookupTable(sineTable, subP) * (0.4f * modAmt);
                    break;
                }

                case FormantChoir:
                {
                    float f1 = lookupTable(formantTableA, cPhase);
                    float f2 = lookupTable(formantTableB, cPhase);
                    sig = f1 + modAmt * (f2 - f1);
                    break;
                }

                case MetallicFM:
                {
                    double ratio = 1.0 + std::floor(modAmt * 4.0) * 0.70710678;
                    double mPhase = modPhase[y] + phaseInc * ratio;
                    mPhase -= std::floor(mPhase);
                    modPhase[y] = mPhase;

                    float modWave = lookupTable(sineTable, mPhase);
                    double fmPhase = cPhase + static_cast<double>(modWave * (0.15f + modAmt * 1.25f));
                    fmPhase -= std::floor(fmPhase);
                    sig = lookupTable(sineTable, fmPhase);
                    break;
                }

                case FilteredNoise:
                {
                    noiseSeed = noiseSeed * 196314165U + 907633515U;
                    float white = (static_cast<float>(noiseSeed) / static_cast<float>(0xFFFFFFFFU)) * 2.0f - 1.0f;
                    float f = juce::jlimit(0.005f, 0.45f, static_cast<float>(2.0 * std::sin(juce::MathConstants<double>::pi * freq * invSR)));
                    float q = 0.08f * (1.0f - modAmt * 0.85f);
                    float hp = white - noiseState1[y] - q * noiseState2[y];
                    float bp = f * hp + noiseState2[y];
                    float lp = f * bp + noiseState1[y];
                    noiseState1[y] = lp;
                    noiseState2[y] = bp;
                    sig = bp * 2.5f;
                    break;
                }

                case SampleGrain:
                {
                    const juce::ScopedTryLock stl(sampleLock);
                    if (stl.isLocked() && hasSample && sampleData.getNumSamples() > 1)
                    {
                        const double pitchRatio = (freq / 130.81278) * (sampleSR * invSR);
                        const int totalSamples = sampleData.getNumSamples();
                        double pos = grainPos[y] + pitchRatio;
                        while (pos >= totalSamples) pos -= totalSamples;
                        while (pos < 0.0) pos += totalSamples;
                        grainPos[y] = pos;

                        int i0 = static_cast<int>(pos);
                        int i1 = (i0 + 1) % totalSamples;
                        float frac = static_cast<float>(pos - i0);
                        const float* ch0 = sampleData.getReadPointer(0);
                        sig = (ch0[i0] + frac * (ch0[i1] - ch0[i0])) * 2.0f;
                    }
                    else
                    {
                        sig = lookupTable(sawTable, cPhase);
                    }
                    break;
                }
            }

            sumL += sig * ampL;
            sumR += sig * ampR;
        }

        constexpr float norm = 1.0f / 14.0f;
        outL = std::tanh(sumL * norm);
        outR = std::tanh(sumR * norm);
    }

private:
    inline float lookupTable(const std::array<float, kTableSize>& table, double normPhase) const
    {
        double idx = normPhase * static_cast<double>(kTableSize);
        int i0 = static_cast<int>(idx) & (kTableSize - 1);
        int i1 = (i0 + 1) & (kTableSize - 1);
        float frac = static_cast<float>(idx - std::floor(idx));
        return table[i0] + frac * (table[i1] - table[i0]);
    }

    void buildWavetables()
    {
        const double twoPi = juce::MathConstants<double>::twoPi;
        for (int i = 0; i < kTableSize; ++i)
        {
            double phase = static_cast<double>(i) / static_cast<double>(kTableSize);
            sineTable[i] = static_cast<float>(std::sin(twoPi * phase));

            double saw = 0.0, sq = 0.0, formA = 0.0, formB = 0.0;
            for (int h = 1; h <= 16; ++h)
            {
                double s = std::sin(twoPi * phase * h);
                double win = std::pow(std::cos((h - 1) * juce::MathConstants<double>::pi / 32.0), 2.0);
                saw += (s / h) * win;
                if (h % 2 == 1)
                    sq += (s / h) * win;

                double envA = std::exp(-0.5 * std::pow((h - 3.0) / 1.5, 2.0)) + 0.5 * std::exp(-0.5 * std::pow((h - 7.0) / 2.0, 2.0));
                formA += s * envA;

                double envB = std::exp(-0.5 * std::pow((h - 2.0) / 1.0, 2.0)) + 0.6 * std::exp(-0.5 * std::pow((h - 10.0) / 2.0, 2.0));
                formB += s * envB;
            }
            sawTable[i]      = static_cast<float>(saw * 0.6);
            squareTable[i]   = static_cast<float>(sq * 0.8);
            formantTableA[i] = static_cast<float>(formA * 0.55);
            formantTableB[i] = static_cast<float>(formB * 0.55);
        }
    }

    double sr = 44100.0;
    std::array<float, kTableSize> sineTable{};
    std::array<float, kTableSize> sawTable{};
    std::array<float, kTableSize> squareTable{};
    std::array<float, kTableSize> formantTableA{};
    std::array<float, kTableSize> formantTableB{};

    std::array<double, kNumRows> carrierPhase{};
    std::array<double, kNumRows> modPhase{};
    std::array<double, kNumRows> grainPos{};
    std::array<float, kNumRows>  noiseState1{};
    std::array<float, kNumRows>  noiseState2{};
    uint32_t noiseSeed = 22222U;

    juce::CriticalSection sampleLock;
    juce::AudioBuffer<float> sampleData;
    double sampleSR = 44100.0;
    bool hasSample = false;
};

//==============================================================================
class MetaCoreProcessor : public juce::AudioProcessor
{
public:
    MetaCoreProcessor()
        : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "Parameters", createParameterLayout()),
          fft(kFFTOrder),
          window(kFFTSize, juce::dsp::WindowingFunction<float>::hann, false)
    {
        formatManager.registerBasicFormats();
        clearCanvas();
        generateHarmonicPreset();
    }

    ~MetaCoreProcessor() override = default;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            "mode", "Room Mode", juce::StringArray{ "Image Synth", "Image Filter" }, 0));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            "synthType", "Oscillator Engine",
            juce::StringArray{ "Pure Sine", "Rich SuperSaw", "Hollow Square", "Formant Choir", "Metallic 2-Op FM", "Resonant Noise", "Sample / Grain" }, 1));

        params.push_back(std::make_unique<juce::AudioParameterBool>(
            "sync", "Sync to Host BPM", true));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "rateBars", "Cycle Length (Bars)",
            juce::NormalisableRange<float>(0.25f, 8.0f, 0.25f), 1.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "rateHz", "Free Rate (Hz)",
            juce::NormalisableRange<float>(0.05f, 10.0f, 0.01f, 0.4f), 0.5f));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            "scale", "Scale Grid",
            juce::StringArray{ "Chromatic", "Major", "Minor Pentatonic", "Whole Tone", "Harmonic Series", "Continuous Log" }, 0));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "basePitch", "Base MIDI Note",
            juce::NormalisableRange<float>(12.0f, 72.0f, 1.0f), 36.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "octaves", "Vertical Range (Octaves)",
            juce::NormalisableRange<float>(1.0f, 8.0f, 0.1f), 5.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "modDepth", "Mod / FM Depth",
            juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.35f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "smooth", "Amp Smoothing",
            juce::NormalisableRange<float>(0.1f, 0.999f, 0.001f), 0.92f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "mix", "Dry/Wet",
            juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 1.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            "gain", "Output Gain (dB)",
            juce::NormalisableRange<float>(-36.0f, 12.0f, 0.1f), 0.0f));

        return { params.begin(), params.end() };
    }

    //==============================================================================
    void prepareToPlay(double sampleRate, int /*samplesPerBlock*/) override
    {
        currentSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
        voiceBank.prepare(currentSampleRate);

        for (int i = 0; i < kCanvasHeight; ++i)
        {
            smoothL[i] = 0.0f;
            smoothR[i] = 0.0f;
            smoothB[i] = 0.0f;
        }

        for (int ch = 0; ch < 2; ++ch)
        {
            inputFifo[ch].assign(kFFTSize, 0.0f);
            outputAccum[ch].assign(kFFTSize * 2, 0.0f);
        }
        fifoWritePos = 0;
        hopCounter = 0;
        updateRowFrequencies();
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override
    {
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        if (!layouts.getMainInputChannelSet().isDisabled() &&
            layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        return true;
    }

    //==============================================================================
    void updateRowFrequencies()
    {
        const int scaleIdx = static_cast<int>(apvts.getRawParameterValue("scale")->load());
        const float baseMidi = apvts.getRawParameterValue("basePitch")->load() + midiPitchOffset;
        const float octaveSpan = apvts.getRawParameterValue("octaves")->load();
        const double baseFreq = 440.0 * std::pow(2.0, (baseMidi - 69.0) / 12.0);

        static const int majorScale[7] = { 0, 2, 4, 5, 7, 9, 11 };
        static const int minPent[5]    = { 0, 3, 5, 7, 10 };

        for (int y = 0; y < kCanvasHeight; ++y)
        {
            const int invY = (kCanvasHeight - 1) - y;
            const double norm = static_cast<double>(invY) / static_cast<double>(kCanvasHeight - 1);
            double freq = baseFreq;

            if (scaleIdx == 0)
            {
                double semitones = std::floor(norm * octaveSpan * 12.0);
                freq = baseFreq * std::pow(2.0, semitones / 12.0);
            }
            else if (scaleIdx == 1)
            {
                int deg = static_cast<int>(norm * octaveSpan * 7.0);
                int oct = deg / 7;
                int semi = oct * 12 + majorScale[deg % 7];
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 2)
            {
                int deg = static_cast<int>(norm * octaveSpan * 5.0);
                int oct = deg / 5;
                int semi = oct * 12 + minPent[deg % 5];
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 3)
            {
                int semi = static_cast<int>(norm * octaveSpan * 6.0) * 2;
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 4)
            {
                freq = baseFreq * static_cast<double>(invY + 1);
            }
            else
            {
                freq = baseFreq * std::pow(2.0, norm * octaveSpan);
            }

            rowFreqs[y] = juce::jlimit(20.0, currentSampleRate * 0.48, freq);
        }
    }

    //==============================================================================
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;

        for (const auto metadata : midiMessages)
        {
            const auto msg = metadata.getMessage();
            if (msg.isNoteOn())
            {
                midiPitchOffset = static_cast<float>(msg.getNoteNumber() - 60);
                playheadPos = 0.0;
            }
        }

        updateRowFrequencies();

        const int mode      = static_cast<int>(apvts.getRawParameterValue("mode")->load());
        const int synthType = static_cast<int>(apvts.getRawParameterValue("synthType")->load());
        const bool sync     = apvts.getRawParameterValue("sync")->load() > 0.5f;
        const float bars    = apvts.getRawParameterValue("rateBars")->load();
        const float rateHz  = apvts.getRawParameterValue("rateHz")->load();
        const float modDep  = apvts.getRawParameterValue("modDepth")->load();
        const float smooth  = apvts.getRawParameterValue("smooth")->load();
        const float mix     = apvts.getRawParameterValue("mix")->load();
        const float gainDb  = apvts.getRawParameterValue("gain")->load();
        const float linGain = juce::Decibels::decibelsToGain(gainDb);

        double cyclesPerSec = static_cast<double>(rateHz);
        if (sync)
        {
            if (auto* ph = getPlayHead())
            {
                if (auto pos = ph->getPosition())
                {
                    double bpm = pos->getBpm().orFallback(120.0);
                    double secondsPerBar = (60.0 / bpm) * 4.0;
                    cyclesPerSec = 1.0 / (secondsPerBar * static_cast<double>(bars));

                    if (pos->getIsPlaying() && pos->getPpqPosition().hasValue())
                    {
                        double ppq = *pos->getPpqPosition();
                        double cyclePpq = static_cast<double>(bars) * 4.0;
                        playheadPos = std::fmod(ppq / cyclePpq, 1.0);
                        if (playheadPos < 0.0) playheadPos += 1.0;
                    }
                }
            }
        }

        const double phaseIncPerSample = cyclesPerSec / currentSampleRate;
        const int numSamples = buffer.getNumSamples();
        const int activeIdx = activeBufferIndex.load(std::memory_order_acquire);
        const auto& grid = canvasBuffers[activeIdx];

        float* outL = buffer.getWritePointer(0);
        float* outR = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : outL;
        const float* inL = buffer.getReadPointer(0);
        const float* inR = buffer.getNumChannels() > 1 ? buffer.getReadPointer(1) : inL;

        for (int s = 0; s < numSamples; ++s)
        {
            const float dryL = inL[s];
            const float dryR = inR[s];

            playheadPos += phaseIncPerSample;
            if (playheadPos >= 1.0)
                playheadPos -= 1.0;

            uiPlayhead.store(static_cast<float>(playheadPos), std::memory_order_relaxed);

            const float exactX = static_cast<float>(playheadPos) * (kCanvasWidth - 1);
            const int x0 = static_cast<int>(exactX) % kCanvasWidth;
            const int x1 = (x0 + 1) % kCanvasWidth;
            const float fracX = exactX - std::floor(exactX);

            float wetL = 0.0f;
            float wetR = 0.0f;

            if (mode == 0)
            {
                const float invSmooth = 1.0f - smooth;
                for (int y = 0; y < kCanvasHeight; ++y)
                {
                    const auto& p0 = grid[x0][y];
                    const auto& p1 = grid[x1][y];
                    const float targetL = p0.r + fracX * (p1.r - p0.r);
                    const float targetR = p0.g + fracX * (p1.g - p0.g);
                    const float targetB = p0.b + fracX * (p1.b - p0.b);

                    smoothL[y] = smooth * smoothL[y] + invSmooth * targetL;
                    smoothR[y] = smooth * smoothR[y] + invSmooth * targetR;
                    smoothB[y] = smooth * smoothB[y] + invSmooth * targetB;
                }

                voiceBank.renderSample(synthType, modDep, rowFreqs.data(),
                                       smoothL.data(), smoothR.data(), smoothB.data(),
                                       wetL, wetR);
            }
            else
            {
                inputFifo[0][fifoWritePos] = dryL;
                inputFifo[1][fifoWritePos] = dryR;

                wetL = outputAccum[0][fifoWritePos];
                wetR = outputAccum[1][fifoWritePos];

                outputAccum[0][fifoWritePos] = 0.0f;
                outputAccum[1][fifoWritePos] = 0.0f;

                fifoWritePos = (fifoWritePos + 1) % kFFTSize;
                if (++hopCounter >= kHopSize)
                {
                    hopCounter = 0;
                    processSpectralFrame(grid, x0, x1, fracX);
                }
            }

            outL[s] = (dryL * (1.0f - mix) + wetL * mix) * linGain;
            outR[s] = (dryR * (1.0f - mix) + wetR * mix) * linGain;
        }
    }

    void processSpectralFrame(const std::array<std::array<PixelRGB, kCanvasHeight>, kCanvasWidth>& grid,
                              int x0, int x1, float fracX)
    {
        std::array<float, kFFTSize * 2> fftBufL{};
        std::array<float, kFFTSize * 2> fftBufR{};

        for (int i = 0; i < kFFTSize; ++i)
        {
            const int idx = (fifoWritePos + i) % kFFTSize;
            fftBufL[i] = inputFifo[0][idx];
            fftBufR[i] = inputFifo[1][idx];
        }

        window.multiplyWithWindowingTable(fftBufL.data(), kFFTSize);
        window.multiplyWithWindowingTable(fftBufR.data(), kFFTSize);

        fft.performRealOnlyForwardTransform(fftBufL.data());
        fft.performRealOnlyForwardTransform(fftBufR.data());

        const int numBins = kFFTSize / 2;
        for (int bin = 1; bin < numBins; ++bin)
        {
            const double binFreq = (static_cast<double>(bin) * currentSampleRate) / static_cast<double>(kFFTSize);

            int bestRow = kCanvasHeight - 1;
            double minDiff = 1e9;
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                double diff = std::abs(rowFreqs[y] - binFreq);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    bestRow = y;
                }
            }

            const auto& p0 = grid[x0][bestRow];
            const auto& p1 = grid[x1][bestRow];
            const float maskL = p0.r + fracX * (p1.r - p0.r);
            const float maskR = p0.g + fracX * (p1.g - p0.g);

            fftBufL[bin * 2]     *= maskL;
            fftBufL[bin * 2 + 1] *= maskL;
            fftBufR[bin * 2]     *= maskR;
            fftBufR[bin * 2 + 1] *= maskR;
        }

        fft.performRealOnlyInverseTransform(fftBufL.data());
        fft.performRealOnlyInverseTransform(fftBufR.data());

        window.multiplyWithWindowingTable(fftBufL.data(), kFFTSize);
        window.multiplyWithWindowingTable(fftBufR.data(), kFFTSize);

        const float scale = 0.5f;
        for (int i = 0; i < kFFTSize; ++i)
        {
            const int idx = (fifoWritePos + i) % kFFTSize;
            outputAccum[0][idx] += fftBufL[i] * scale;
            outputAccum[1][idx] += fftBufR[i] * scale;
        }
    }

    //==============================================================================
    void paintBrush(int cx, int cy, int radius, float rWeight, float gWeight, float bWeight, bool erase)
    {
        const int active = activeBufferIndex.load(std::memory_order_acquire);
        const int back   = 1 - active;
        canvasBuffers[back] = canvasBuffers[active];

        for (int dx = -radius; dx <= radius; ++dx)
        {
            for (int dy = -radius; dy <= radius; ++dy)
            {
                if (dx * dx + dy * dy <= radius * radius)
                {
                    int x = juce::jlimit(0, kCanvasWidth - 1, cx + dx);
                    int y = juce::jlimit(0, kCanvasHeight - 1, cy + dy);
                    float falloff = 1.0f - (std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(radius + 1));

                    if (erase)
                    {
                        canvasBuffers[back][x][y] = { 0.0f, 0.0f, 0.0f };
                    }
                    else
                    {
                        canvasBuffers[back][x][y].r = juce::jlimit(0.0f, 1.0f, canvasBuffers[back][x][y].r + rWeight * falloff);
                        canvasBuffers[back][x][y].g = juce::jlimit(0.0f, 1.0f, canvasBuffers[back][x][y].g + gWeight * falloff);
                        canvasBuffers[back][x][y].b = juce::jlimit(0.0f, 1.0f, canvasBuffers[back][x][y].b + bWeight * falloff);
                    }
                }
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void clearCanvas()
    {
        const int back = 1 - activeBufferIndex.load();
        for (int x = 0; x < kCanvasWidth; ++x)
            for (int y = 0; y < kCanvasHeight; ++y)
                canvasBuffers[back][x][y] = { 0.0f, 0.0f, 0.0f };
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void generateHarmonicPreset()
    {
        const int back = 1 - activeBufferIndex.load();
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                canvasBuffers[back][x][y] = { 0.0f, 0.0f, 0.0f };
                int invY = (kCanvasHeight - 1) - y;
                if (invY % 12 == 0 || invY % 19 == 0)
                {
                    float phase = (static_cast<float>(x) / kCanvasWidth) * juce::MathConstants<float>::twoPi * 2.0f;
                    float env = 0.5f + 0.5f * std::sin(phase + invY * 0.1f);
                    float modWave = 0.5f + 0.5f * std::cos(phase * 1.5f);
                    canvasBuffers[back][x][y].r = env * ((invY % 24 == 0) ? 0.9f : 0.35f);
                    canvasBuffers[back][x][y].g = (1.0f - env) * 0.85f;
                    canvasBuffers[back][x][y].b = modWave * 0.7f;
                }
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void blurHorizontal()
    {
        const int active = activeBufferIndex.load();
        const int back = 1 - active;
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            int xm1 = (x - 1 + kCanvasWidth) % kCanvasWidth;
            int xp1 = (x + 1) % kCanvasWidth;
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                canvasBuffers[back][x][y].r = 0.25f * canvasBuffers[active][xm1][y].r +
                                              0.50f * canvasBuffers[active][x][y].r +
                                              0.25f * canvasBuffers[active][xp1][y].r;
                canvasBuffers[back][x][y].g = 0.25f * canvasBuffers[active][xm1][y].g +
                                              0.50f * canvasBuffers[active][x][y].g +
                                              0.25f * canvasBuffers[active][xp1][y].g;
                canvasBuffers[back][x][y].b = 0.25f * canvasBuffers[active][xm1][y].b +
                                              0.50f * canvasBuffers[active][x][y].b +
                                              0.25f * canvasBuffers[active][xp1][y].b;
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void invertCanvas()
    {
        const int active = activeBufferIndex.load();
        const int back = 1 - active;
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                canvasBuffers[back][x][y].r = 1.0f - canvasBuffers[active][x][y].r;
                canvasBuffers[back][x][y].g = 1.0f - canvasBuffers[active][x][y].g;
                canvasBuffers[back][x][y].b = 1.0f - canvasBuffers[active][x][y].b;
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void loadFromImage(const juce::Image& img)
    {
        if (!img.isValid()) return;
        const int back = 1 - activeBufferIndex.load();
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            int sx = juce::jlimit(0, img.getWidth() - 1, (x * img.getWidth()) / kCanvasWidth);
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                int sy = juce::jlimit(0, img.getHeight() - 1, (y * img.getHeight()) / kCanvasHeight);
                juce::Colour c = img.getPixelAt(sx, sy);
                canvasBuffers[back][x][y] = { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue() };
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void loadAudioSampleFile(const juce::File& file)
    {
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
        if (reader != nullptr)
        {
            const int maxSamples = static_cast<int>(juce::jmin<juce::int64>(reader->lengthInSamples, 44100 * 10));
            juce::AudioBuffer<float> tempBuf(1, maxSamples);
            reader->read(&tempBuf, 0, maxSamples, 0, true, false);
            voiceBank.loadSampleBuffer(tempBuf, reader->sampleRate);
        }
    }

    const std::array<std::array<PixelRGB, kCanvasHeight>, kCanvasWidth>& getActiveCanvas() const
    {
        return canvasBuffers[activeBufferIndex.load(std::memory_order_acquire)];
    }

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "MetaCore"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());

        juce::MemoryBlock canvasData;
        const auto& grid = getActiveCanvas();
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                uint8_t r8 = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].r * 255.0f));
                uint8_t g8 = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].g * 255.0f));
                uint8_t b8 = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].b * 255.0f));
                canvasData.append(&r8, 1);
                canvasData.append(&g8, 1);
                canvasData.append(&b8, 1);
            }
        }
        xml->setAttribute("canvasRGBBase64", canvasData.toBase64Encoding());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
        if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
        {
            apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
            if (xmlState->hasAttribute("canvasRGBBase64"))
            {
                juce::MemoryBlock canvasData;
                if (canvasData.fromBase64Encoding(xmlState->getStringAttribute("canvasRGBBase64")))
                {
                    if (canvasData.getSize() == kCanvasWidth * kCanvasHeight * 3)
                    {
                        const auto* bytes = static_cast<const uint8_t*>(canvasData.getData());
                        const int back = 1 - activeBufferIndex.load();
                        size_t idx = 0;
                        for (int x = 0; x < kCanvasWidth; ++x)
                        {
                            for (int y = 0; y < kCanvasHeight; ++y)
                            {
                                canvasBuffers[back][x][y].r = bytes[idx++] / 255.0f;
                                canvasBuffers[back][x][y].g = bytes[idx++] / 255.0f;
                                canvasBuffers[back][x][y].b = bytes[idx++] / 255.0f;
                            }
                        }
                        activeBufferIndex.store(back, std::memory_order_release);
                    }
                }
            }
        }
    }

    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float> uiPlayhead{ 0.0f };

private:
    double currentSampleRate = 44100.0;
    double playheadPos = 0.0;
    float midiPitchOffset = 0.0f;

    MetaVoiceBank voiceBank;
    juce::AudioFormatManager formatManager;

    std::array<std::array<std::array<PixelRGB, kCanvasHeight>, kCanvasWidth>, 2> canvasBuffers{};
    std::atomic<int> activeBufferIndex{ 0 };

    std::array<double, kCanvasHeight> rowFreqs{};
    std::array<float, kCanvasHeight> smoothL{};
    std::array<float, kCanvasHeight> smoothR{};
    std::array<float, kCanvasHeight> smoothB{};

    juce::dsp::FFT fft;
    juce::dsp::WindowingFunction<float> window;
    std::array<std::vector<float>, 2> inputFifo;
    std::array<std::vector<float>, 2> outputAccum;
    int fifoWritePos = 0;
    int hopCounter = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MetaCoreProcessor)
};

//==============================================================================
class MetaCanvasComponent : public juce::Component, private juce::Timer
{
public:
    explicit MetaCanvasComponent(MetaCoreProcessor& p)
        : processor(p), displayImage(juce::Image::RGB, kCanvasWidth, kCanvasHeight, true)
    {
        startTimerHz(30);
    }

    void setBrushColorMode(int mode) { colorMode = mode; }
    void setBrushSize(int r) { brushRadius = r; }

    void mouseDown(const juce::MouseEvent& e) override { paintAt(e); }
    void mouseDrag(const juce::MouseEvent& e) override { paintAt(e); }

    void paintAt(const juce::MouseEvent& e)
    {
        int cx = (e.x * kCanvasWidth) / getWidth();
        int cy = (e.y * kCanvasHeight) / getHeight();
        bool erase = e.mods.isRightButtonDown() || (colorMode == 5);
        float rW = (colorMode == 0 || colorMode == 1 || colorMode == 4) ? 0.35f : 0.0f;
        float gW = (colorMode == 0 || colorMode == 2 || colorMode == 4) ? 0.35f : 0.0f;
        float bW = (colorMode == 3 || colorMode == 4) ? 0.45f : 0.0f;
        processor.paintBrush(cx, cy, brushRadius, rW, gW, bW, erase);
        rebuildImage();
        repaint();
    }

    void rebuildImage()
    {
        const auto& grid = processor.getActiveCanvas();
        juce::Image::BitmapData bd(displayImage, juce::Image::BitmapData::writeOnly);
        for (int y = 0; y < kCanvasHeight; ++y)
        {
            for (int x = 0; x < kCanvasWidth; ++x)
            {
                uint8_t r = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].r * 255.0f));
                uint8_t g = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].g * 255.0f));
                uint8_t gridLine = static_cast<uint8_t>((y % 12 == 0) ? 28 : 8);
                uint8_t b = static_cast<uint8_t>(juce::jlimit<int>(gridLine, 255, static_cast<int>(grid[x][y].b * 255.0f)));
                bd.setPixelColour(x, y, juce::Colour(r, g, b));
            }
        }
    }

    void paint(juce::Graphics& g) override
    {
        g.drawImage(displayImage, getLocalBounds().toFloat(), juce::RectanglePlacement::stretchToFit);

        float ph = processor.uiPlayhead.load(std::memory_order_relaxed);
        float px = ph * static_cast<float>(getWidth());
        g.setColour(juce::Colours::cyan.withAlpha(0.85f));
        g.drawVerticalLine(static_cast<int>(px), 0.0f, static_cast<float>(getHeight()));

        g.setColour(juce::Colours::white.withAlpha(0.25f));
        g.drawRect(getLocalBounds(), 1);
    }

private:
    void timerCallback() override
    {
        rebuildImage();
        repaint();
    }

    MetaCoreProcessor& processor;
    juce::Image displayImage;
    int colorMode = 0;
    int brushRadius = 2;
};

//==============================================================================
class MetaCoreEditor : public juce::AudioProcessorEditor
{
public:
    explicit MetaCoreEditor(MetaCoreProcessor& p)
        : AudioProcessorEditor(&p), processor(p), canvas(p)
    {
        setSize(960, 580);
        addAndMakeVisible(canvas);

        setupCombo(modeBox, "mode", { "Image Synth", "Image Filter" });
        setupCombo(synthBox, "synthType", { "Pure Sine", "Rich SuperSaw", "Hollow Square", "Formant Choir", "Metallic 2-Op FM", "Resonant Noise", "Sample / Grain" });
        setupCombo(scaleBox, "scale", { "Chromatic", "Major", "Minor Pentatonic", "Whole Tone", "Harmonic Series", "Continuous Log" });

        addAndMakeVisible(brushBox);
        brushBox.addItemList({
            "Brush: Yellow (Center)",
            "Brush: Red (Left)",
            "Brush: Green (Right)",
            "Brush: Blue (FM/Mod)",
            "Brush: White (L+R+Mod)",
            "Eraser"
        }, 1);
        brushBox.setSelectedId(1);
        brushBox.onChange = [this] { canvas.setBrushColorMode(brushBox.getSelectedId() - 1); };

        addAndMakeVisible(sizeBox);
        sizeBox.addItemList({ "Size: 1px", "Size: 3px", "Size: 6px", "Size: 12px" }, 1);
        sizeBox.setSelectedId(2);
        sizeBox.onChange = [this] {
            const int sizes[] = { 0, 2, 5, 11 };
            canvas.setBrushSize(sizes[juce::jlimit(0, 3, sizeBox.getSelectedId() - 1)]);
        };

        setupButton(loadImgBtn, "Load Image...", [this] { promptLoadImage(); });
        setupButton(loadWavBtn, "Load Sample WAV...", [this] { promptLoadSample(); });
        setupButton(presetBtn, "Harmonic Grid", [this] { processor.generateHarmonicPreset(); canvas.rebuildImage(); });
        setupButton(blurBtn, "Smooth X", [this] { processor.blurHorizontal(); canvas.rebuildImage(); });
        setupButton(invertBtn, "Invert", [this] { processor.invertCanvas(); canvas.rebuildImage(); });
        setupButton(clearBtn, "Clear", [this] { processor.clearCanvas(); canvas.rebuildImage(); });

        addAndMakeVisible(syncToggle);
        syncToggle.setButtonText("Sync BPM");
        syncAttach = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(processor.apvts, "sync", syncToggle);

        setupKnob(barsKnob, barsLabel, "rateBars", "Bars");
        setupKnob(hzKnob, hzLabel, "rateHz", "Free Hz");
        setupKnob(pitchKnob, pitchLabel, "basePitch", "Base Note");
        setupKnob(octKnob, octLabel, "octaves", "Octaves");
        setupKnob(modKnob, modLabel, "modDepth", "Mod / FM");
        setupKnob(smoothKnob, smoothLabel, "smooth", "Smoothing");
        setupKnob(mixKnob, mixLabel, "mix", "Dry/Wet");
        setupKnob(gainKnob, gainLabel, "gain", "Gain dB");
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff141619));
        g.setColour(juce::Colours::orange);
        g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        g.drawText("METACORE // PHASE 1 MULTI-ENGINE SYNTH & FILTER", 16, 8, 460, 28, juce::Justification::centredLeft);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        auto topBar = area.removeFromTop(32);
        topBar.removeFromLeft(360);
        modeBox.setBounds(topBar.removeFromLeft(130).reduced(2));
        synthBox.setBounds(topBar.removeFromLeft(155).reduced(2));
        scaleBox.setBounds(topBar.removeFromLeft(155).reduced(2));
        syncToggle.setBounds(topBar.removeFromLeft(100).reduced(2));

        area.removeFromTop(8);
        auto toolBar = area.removeFromTop(30);
        brushBox.setBounds(toolBar.removeFromLeft(175).reduced(2));
        sizeBox.setBounds(toolBar.removeFromLeft(100).reduced(2));
        loadImgBtn.setBounds(toolBar.removeFromLeft(110).reduced(2));
        loadWavBtn.setBounds(toolBar.removeFromLeft(135).reduced(2));
        presetBtn.setBounds(toolBar.removeFromLeft(110).reduced(2));
        blurBtn.setBounds(toolBar.removeFromLeft(90).reduced(2));
        invertBtn.setBounds(toolBar.removeFromLeft(80).reduced(2));
        clearBtn.setBounds(toolBar.removeFromLeft(80).reduced(2));

        area.removeFromTop(8);
        auto bottomKnobs = area.removeFromBottom(95);
        canvas.setBounds(area);

        int knobW = bottomKnobs.getWidth() / 8;
        layoutKnob(barsKnob, barsLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(hzKnob, hzLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(pitchKnob, pitchLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(octKnob, octLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(modKnob, modLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(smoothKnob, smoothLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(mixKnob, mixLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(gainKnob, gainLabel, bottomKnobs.removeFromLeft(knobW));
    }

private:
    void setupCombo(juce::ComboBox& box, const juce::String& paramID, const juce::StringArray& items)
    {
        addAndMakeVisible(box);
        box.addItemList(items, 1);
        comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(processor.apvts, paramID, box));
    }

    void setupButton(juce::TextButton& btn, const juce::String& text, std::function<void()> fn)
    {
        addAndMakeVisible(btn);
        btn.setButtonText(text);
        btn.onClick = std::move(fn);
    }

    void setupKnob(juce::Slider& s, juce::Label& l, const juce::String& paramID, const juce::String& name)
    {
        addAndMakeVisible(s);
        s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 16);
        addAndMakeVisible(l);
        l.setText(name, juce::dontSendNotification);
        l.setJustificationType(juce::Justification::centred);
        sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor.apvts, paramID, s));
    }

    void layoutKnob(juce::Slider& s, juce::Label& l, juce::Rectangle<int> r)
    {
        l.setBounds(r.removeFromTop(18));
        s.setBounds(r);
    }

    void promptLoadImage()
    {
        fileChooser = std::make_unique<juce::FileChooser>("Select an Image (PNG/JPG)", juce::File{}, "*.png;*.jpg;*.jpeg;*.bmp");
        fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file.existsAsFile())
                {
                    juce::Image img = juce::ImageFileFormat::loadFrom(file);
                    processor.loadFromImage(img);
                    canvas.rebuildImage();
                }
            });
    }

    void promptLoadSample()
    {
        fileChooser = std::make_unique<juce::FileChooser>("Select a WAV/AIFF Sample for Grain/Sample Synth", juce::File{}, "*.wav;*.aif;*.aiff;*.flac");
        fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file.existsAsFile())
                {
                    processor.loadAudioSampleFile(file);
                }
            });
    }

    MetaCoreProcessor& processor;
    MetaCanvasComponent canvas;

    juce::ComboBox modeBox, synthBox, scaleBox, brushBox, sizeBox;
    juce::TextButton loadImgBtn, loadWavBtn, presetBtn, blurBtn, invertBtn, clearBtn;
    juce::ToggleButton syncToggle;

    juce::Slider barsKnob, hzKnob, pitchKnob, octKnob, modKnob, smoothKnob, mixKnob, gainKnob;
    juce::Label barsLabel, hzLabel, pitchLabel, octLabel, modLabel, smoothLabel, mixLabel, gainLabel;

    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> syncAttach;
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MetaCoreEditor)
};

juce::AudioProcessorEditor* MetaCoreProcessor::createEditor()
{
    return new MetaCoreEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MetaCoreProcessor();
}
