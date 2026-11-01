#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <array>
#include <vector>
#include <cmath>
#include <random>

struct PixelRGB
{
    float r = 0.0f; // Left channel amplitude
    float g = 0.0f; // Right channel amplitude
    float b = 0.0f; // Modulation depth (FM index / Harmonic morph / Detune)
};

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
            // Randomize initial phases slightly to prevent massive click on playhead start
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

    bool hasLoadedSample() const { return hasSample; }

    // Renders a single stereo sample across all 128 rows
    inline void renderSample(int synthType,
                             float modDepthKnob,
                             doublerowFreqs,
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

            // Advance phase even if silent so partials stay phase-coherent
            double cPhase = carrierPhase[y] + phaseInc;
            cPhase -= std::floor(cPhase);
            carrierPhase[y] = cPhase;

            if (ampL <= 0.0001f && ampR <= 0.0001f)
                continue;

            // Effective modulation combines the global Mod knob with per-pixel Blue intensity
            const float modAmt = juce::jlimit(0.0f, 1.0f, smoothB[y] + modDepthKnob * 0.5f);
            float sig = 0.0f;

            switch (synthType)
            {
                case Sine:
                {
                    // Blue channel adds subtle 2nd/3rd harmonic saturation
                    float fundamental = lookupTable(sineTable, cPhase);
                    if (modAmt > 0.01f)
                    {
                        double p2 = cPhase * 2.0; p2 -= std::floor(p2);
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
                    // Morphs between warm saw and bright super-saw
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
                    // Dual-formant table morph driven by Blue channel
                    float f1 = lookupTable(formantTableA, cPhase);
                    float f2 = lookupTable(formantTableB, cPhase);
                    sig = f1 + modAmt * (f2 - f1);
                    break;
                }

                case MetallicFM:
                {
                    // Classic 2-Op FM with non-integer metallic ratio (1 : 1.4142 or 1 : 2.0)
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
                    // Bandpass-resonated noise tuned to row frequency
                    noiseSeed = noiseSeed * 196314165U + 907633515U;
                    float white = (static_cast<float>(noiseSeed) / static_cast<float>(0xFFFFFFFFU)) * 2.0f - 1.0f;
                    float f = juce::jlimit(0.005f, 0.45f, static_cast<float>(2.0 * std::sin(juce::MathConstants<double>::pi * freq * invSR)));
                    float q = 0.08f * (1.0f - modAmt * 0.85f); // Blue channel narrows bandwidth into whistling sine
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
                    if (hasSample && sampleData.getNumSamples() > 1)
                    {
                        // Base root is C3 (130.81 Hz); pitch-shift sample per row
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

        // Normalize 128-voice sum cleanly with soft-clipping protection
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

            // Band-limited additive tables (16 harmonics)
            double saw = 0.0, sq = 0.0, formA = 0.0, formB = 0.0;
            for (int h = 1; h <= 16; ++h)
            {
                double s = std::sin(twoPi * phase * h);
                // Gibbs suppression window
                double win = std::pow(std::cos((h - 1) * juce::MathConstants<double>::pi / 32.0), 2.0);
                saw += (s / h) * win;
                if (h % 2 == 1)
                    sq += (s / h) * win;

                // Formant A: peaks around 3rd and 7th harmonics (Vowel "Ah")
                double envA = std::exp(-0.5 * std::pow((h - 3.0) / 1.5, 2.0)) + 0.5 * std::exp(-0.5 * std::pow((h - 7.0) / 2.0, 2.0));
                formA += s * envA;

                // Formant B: peaks around 2nd and 10th harmonics (Vowel "Ee")
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
