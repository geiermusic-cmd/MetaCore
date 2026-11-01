#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <atomic>
#include <vector>
#include <array>
#include <cmath>

static constexpr int kCanvasWidth  = 256;
static constexpr int kCanvasHeight = 128;
static constexpr int kFFTOrder     = 10; // 1024-point FFT
static constexpr int kFFTSize      = 1 << kFFTOrder;
static constexpr int kHopSize      = kFFTSize / 4; // 4x overlap

struct PixelRG
{
    float r = 0.0f; // Left channel weight
    float g = 0.0f; // Right channel weight
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
        clearCanvas();
        generateHarmonicPreset();
    }

    ~MetaCoreProcessor() override = default;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            "mode", "Room Mode", juce::StringArray{ "Image Synth", "Image Filter" }, 0));

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
        for (int i = 0; i < kCanvasHeight; ++i)
        {
            phases[i] = 0.0;
            smoothL[i] = 0.0f;
            smoothR[i] = 0.0f;
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
            // Row 0 is top (highest pitch), Row 127 is bottom (lowest pitch)
            const int invY = (kCanvasHeight - 1) - y;
            const double norm = static_cast<double>(invY) / static_cast<double>(kCanvasHeight - 1);
            double freq = baseFreq;

            if (scaleIdx == 0) // Chromatic
            {
                double semitones = std::floor(norm * octaveSpan * 12.0);
                freq = baseFreq * std::pow(2.0, semitones / 12.0);
            }
            else if (scaleIdx == 1) // Major
            {
                int deg = static_cast<int>(norm * octaveSpan * 7.0);
                int oct = deg / 7;
                int semi = oct * 12 + majorScale[deg % 7];
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 2) // Minor Pentatonic
            {
                int deg = static_cast<int>(norm * octaveSpan * 5.0);
                int oct = deg / 5;
                int semi = oct * 12 + minPent[deg % 5];
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 3) // Whole Tone
            {
                int semi = static_cast<int>(norm * octaveSpan * 6.0) * 2;
                freq = baseFreq * std::pow(2.0, semi / 12.0);
            }
            else if (scaleIdx == 4) // Harmonic Series
            {
                freq = baseFreq * static_cast<double>(invY + 1);
            }
            else // Continuous Log
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

        // Parse MIDI for pitch transposition or playhead retrigger
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
        const bool sync     = apvts.getRawParameterValue("sync")->load() > 0.5f;
        const float bars    = apvts.getRawParameterValue("rateBars")->load();
        const float rateHz  = apvts.getRawParameterValue("rateHz")->load();
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

        const double twoPi = juce::MathConstants<double>::twoPi;
        const float normFactor = 1.0f / 16.0f; // Prevent oscillator bank summing overload

        for (int s = 0; s < numSamples; ++s)
        {
            const float dryL = inL[s];
            const float dryR = inR[s];

            playheadPos += phaseIncPerSample;
            if (playheadPos >= 1.0)
                playheadPos -= 1.0;

            uiPlayhead.store(static_cast<float>(playheadPos), std::memory_order_relaxed);

            // Bilinear horizontal interpolation across columns
            const float exactX = static_cast<float>(playheadPos) * (kCanvasWidth - 1);
            const int x0 = static_cast<int>(exactX) % kCanvasWidth;
            const int x1 = (x0 + 1) % kCanvasWidth;
            const float fracX = exactX - std::floor(exactX);

            float wetL = 0.0f;
            float wetR = 0.0f;

            if (mode == 0) // IMAGE SYNTH ROOM (Oscillator Bank)
            {
                for (int y = 0; y < kCanvasHeight; ++y)
                {
                    const auto& p0 = grid[x0][y];
                    const auto& p1 = grid[x1][y];
                    const float targetL = p0.r + fracX * (p1.r - p0.r);
                    const float targetR = p0.g + fracX * (p1.g - p0.g);

                    smoothL[y] = smooth * smoothL[y] + (1.0f - smooth) * targetL;
                    smoothR[y] = smooth * smoothR[y] + (1.0f - smooth) * targetR;

                    if (smoothL[y] > 0.0001f || smoothR[y] > 0.0001f)
                    {
                        const float osc = static_cast<float>(std::sin(phases[y]));
                        wetL += osc * smoothL[y];
                        wetR += osc * smoothR[y];
                    }

                    phases[y] += twoPi * rowFreqs[y] / currentSampleRate;
                    if (phases[y] >= twoPi)
                        phases[y] -= twoPi;
                }
                wetL *= normFactor;
                wetR *= normFactor;
            }
            else // IMAGE FILTER ROOM (STFT Spectral Masking)
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

    void processSpectralFrame(const std::array<std::array<PixelRG, kCanvasHeight>, kCanvasWidth>& grid,
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

        // Map FFT bins to nearest canvas row frequency
        const int numBins = kFFTSize / 2;
        for (int bin = 1; bin < numBins; ++bin)
        {
            const double binFreq = (static_cast<double>(bin) * currentSampleRate) / static_cast<double>(kFFTSize);

            // Find closest row in rowFreqs (row 0 is highest, row 127 is lowest)
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

        const float scale = 0.5f; // Hann 4x overlap-add compensation
        for (int i = 0; i < kFFTSize; ++i)
        {
            const int idx = (fifoWritePos + i) % kFFTSize;
            outputAccum[0][idx] += fftBufL[i] * scale;
            outputAccum[1][idx] += fftBufR[i] * scale;
        }
    }

    //==============================================================================
    // Canvas Manipulation Methods (Thread-safe double buffering)
    void paintBrush(int cx, int cy, int radius, float rWeight, float gWeight, bool erase)
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
                        canvasBuffers[back][x][y].r = 0.0f;
                        canvasBuffers[back][x][y].g = 0.0f;
                    }
                    else
                    {
                        canvasBuffers[back][x][y].r = juce::jlimit(0.0f, 1.0f, canvasBuffers[back][x][y].r + rWeight * falloff);
                        canvasBuffers[back][x][y].g = juce::jlimit(0.0f, 1.0f, canvasBuffers[back][x][y].g + gWeight * falloff);
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
                canvasBuffers[back][x][y] = { 0.0f, 0.0f };
        activeBufferIndex.store(back, std::memory_order_release);
    }

    void generateHarmonicPreset()
    {
        const int back = 1 - activeBufferIndex.load();
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                canvasBuffers[back][x][y] = { 0.0f, 0.0f };
                // Draw sweeping stereo arpeggiated harmonics
                int invY = (kCanvasHeight - 1) - y;
                if (invY % 12 == 0 || invY % 19 == 0)
                {
                    float env = 0.5f + 0.5f * std::sin((static_cast<float>(x) / kCanvasWidth) * juce::MathConstants<float>::twoPi * 2.0f + invY * 0.1f);
                    canvasBuffers[back][x][y].r = env * ((invY % 24 == 0) ? 0.9f : 0.3f);
                    canvasBuffers[back][x][y].g = (1.0f - env) * 0.8f;
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
                // MetaSynth convention: Red = Left, Green = Right, Blue ignored (or mapped to both if grayscale)
                float r = c.getFloatRed();
                float g = c.getFloatGreen();
                canvasBuffers[back][x][y] = { r, g };
            }
        }
        activeBufferIndex.store(back, std::memory_order_release);
    }

    const std::array<std::array<PixelRG, kCanvasHeight>, kCanvasWidth>& getActiveCanvas() const
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

        // Serialize canvas pixels into binary base64 attribute
        juce::MemoryBlock canvasData;
        const auto& grid = getActiveCanvas();
        for (int x = 0; x < kCanvasWidth; ++x)
        {
            for (int y = 0; y < kCanvasHeight; ++y)
            {
                uint8_t r8 = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].r * 255.0f));
                uint8_t g8 = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, grid[x][y].g * 255.0f));
                canvasData.append(&r8, 1);
                canvasData.append(&g8, 1);
            }
        }
        xml->setAttribute("canvasBase64", canvasData.toBase64Encoding());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
        if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
        {
            apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
            if (xmlState->hasAttribute("canvasBase64"))
            {
                juce::MemoryBlock canvasData;
                if (canvasData.fromBase64Encoding(xmlState->getStringAttribute("canvasBase64")))
                {
                    if (canvasData.getSize() == kCanvasWidth * kCanvasHeight * 2)
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

    std::array<std::array<std::array<PixelRG, kCanvasHeight>, kCanvasWidth>, 2> canvasBuffers{};
    std::atomic<int> activeBufferIndex{ 0 };

    std::array<double, kCanvasHeight> rowFreqs{};
    std::array<double, kCanvasHeight> phases{};
    std::array<float, kCanvasHeight> smoothL{};
    std::array<float, kCanvasHeight> smoothR{};

    // STFT Filter Room members
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
    explicit MetaCanvasComponent(MetaCoreProcessor& p) : processor(p), displayImage(juce::Image::RGB, kCanvasWidth, kCanvasHeight, true)
    {
        startTimerHz(30);
    }

    void setBrushColorMode(int mode) { colorMode = mode; } // 0: Yellow(C), 1: Red(L), 2: Green(R), 3: Erase
    void setBrushSize(int r) { brushRadius = r; }

    void mouseDown(const juce::MouseEvent& e) override { paintAt(e); }
    void mouseDrag(const juce::MouseEvent& e) override { paintAt(e); }

    void paintAt(const juce::MouseEvent& e)
    {
        int cx = (e.x * kCanvasWidth) / getWidth();
        int cy = (e.y * kCanvasHeight) / getHeight();
        bool erase = e.mods.isRightButtonDown() || (colorMode == 3);
        float rW = (colorMode == 0 || colorMode == 1) ? 0.35f : 0.0f;
        float gW = (colorMode == 0 || colorMode == 2) ? 0.35f : 0.0f;
        processor.paintBrush(cx, cy, brushRadius, rW, gW, erase);
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
                uint8_t b = static_cast<uint8_t>((y % 12 == 0) ? 28 : 10); // Subtle octave grid lines
                bd.setPixelColour(x, y, juce::Colour(r, g, b));
            }
        }
    }

    void paint(juce::Graphics& g) override
    {
        g.drawImage(displayImage, getLocalBounds().toFloat(), juce::RectanglePlacement::stretchToFit);

        // Draw Playhead
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
        setSize(900, 560);
        addAndMakeVisible(canvas);

        // Room & Scale Selectors
        setupCombo(modeBox, "mode", { "Image Synth", "Image Filter" });
        setupCombo(scaleBox, "scale", { "Chromatic", "Major", "Minor Pentatonic", "Whole Tone", "Harmonic Series", "Continuous Log" });

        // Brush Toolbar
        addAndMakeVisible(brushBox);
        brushBox.addItemList({ "Brush: Yellow (Center)", "Brush: Red (Left)", "Brush: Green (Right)", "Eraser" }, 1);
        brushBox.setSelectedId(1);
        brushBox.onChange = [this] { canvas.setBrushColorMode(brushBox.getSelectedId() - 1); };

        addAndMakeVisible(sizeBox);
        sizeBox.addItemList({ "Size: 1px", "Size: 3px", "Size: 6px", "Size: 12px" }, 1);
        sizeBox.setSelectedId(2);
        sizeBox.onChange = [this] {
            const int sizes[] = { 0, 2, 5, 11 };
            canvas.setBrushSize(sizes[juce::jlimit(0, 3, sizeBox.getSelectedId() - 1)]);
        };

        // Action Buttons
        setupButton(loadImgBtn, "Load Image...", [this] { promptLoadImage(); });
        setupButton(presetBtn, "Harmonic Grid", [this] { processor.generateHarmonicPreset(); canvas.rebuildImage(); });
        setupButton(blurBtn, "Smooth X", [this] { processor.blurHorizontal(); canvas.rebuildImage(); });
        setupButton(invertBtn, "Invert", [this] { processor.invertCanvas(); canvas.rebuildImage(); });
        setupButton(clearBtn, "Clear", [this] { processor.clearCanvas(); canvas.rebuildImage(); });

        addAndMakeVisible(syncToggle);
        syncToggle.setButtonText("Sync BPM");
        syncAttach = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(processor.apvts, "sync", syncToggle);

        // Rotary Knobs
        setupKnob(barsKnob, barsLabel, "rateBars", "Bars");
        setupKnob(hzKnob, hzLabel, "rateHz", "Free Hz");
        setupKnob(pitchKnob, pitchLabel, "basePitch", "Base Note");
        setupKnob(octKnob, octLabel, "octaves", "Octaves");
        setupKnob(smoothKnob, smoothLabel, "smooth", "Smoothing");
        setupKnob(mixKnob, mixLabel, "mix", "Dry/Wet");
        setupKnob(gainKnob, gainLabel, "gain", "Gain dB");
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff141619));
        g.setColour(juce::Colours::orange);
        g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        g.drawText("METACORE // SPECTRAL IMAGE SYNTH & FILTER", 16, 8, 450, 28, juce::Justification::centredLeft);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        auto topBar = area.removeFromTop(32);
        topBar.removeFromLeft(420); // Title space
        modeBox.setBounds(topBar.removeFromLeft(140).reduced(2));
        scaleBox.setBounds(topBar.removeFromLeft(160).reduced(2));
        syncToggle.setBounds(topBar.removeFromLeft(110).reduced(2));

        area.removeFromTop(8);
        auto toolBar = area.removeFromTop(30);
        brushBox.setBounds(toolBar.removeFromLeft(170).reduced(2));
        sizeBox.setBounds(toolBar.removeFromLeft(110).reduced(2));
        loadImgBtn.setBounds(toolBar.removeFromLeft(110).reduced(2));
        presetBtn.setBounds(toolBar.removeFromLeft(110).reduced(2));
        blurBtn.setBounds(toolBar.removeFromLeft(90).reduced(2));
        invertBtn.setBounds(toolBar.removeFromLeft(80).reduced(2));
        clearBtn.setBounds(toolBar.removeFromLeft(80).reduced(2));

        area.removeFromTop(8);
        auto bottomKnobs = area.removeFromBottom(95);
        canvas.setBounds(area);

        int knobW = bottomKnobs.getWidth() / 7;
        layoutKnob(barsKnob, barsLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(hzKnob, hzLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(pitchKnob, pitchLabel, bottomKnobs.removeFromLeft(knobW));
        layoutKnob(octKnob, octLabel, bottomKnobs.removeFromLeft(knobW));
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

    MetaCoreProcessor& processor;
    MetaCanvasComponent canvas;

    juce::ComboBox modeBox, scaleBox, brushBox, sizeBox;
    juce::TextButton loadImgBtn, presetBtn, blurBtn, invertBtn, clearBtn;
    juce::ToggleButton syncToggle;

    juce::Slider barsKnob, hzKnob, pitchKnob, octKnob, smoothKnob, mixKnob, gainKnob;
    juce::Label barsLabel, hzLabel, pitchLabel, octLabel, smoothLabel, mixLabel, gainLabel;

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
