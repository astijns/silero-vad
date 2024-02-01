#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string>
#include "onnxruntime_cxx_api.h"
#include "wav.h"
#include <cstdio>
#include <cstdarg>
#include <fmt/format.h>

#if __cplusplus < 201703L
#include <memory>
#endif

//#define __DEBUG_SPEECH_PROB___

using namespace std::chrono_literals;

enum speechState {
    On,
    Off,
    MaybeOn,
    MaybeOff
};

class timestamp_t {
public:
    int start;
    int end;

    // default + parameterized constructor
    explicit timestamp_t(int sampleRateMs, int start = -1, int end = -1)
            : sampleRateMs_{sampleRateMs}, start{start}, end{end} {
    };

    // assignment operator modifies object, therefore non-const
    timestamp_t &operator=(const timestamp_t &a) = default;

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t &a) const {
        return (start == a.start && end == a.end);
    };

    std::string c_str() {
        auto startDuration = std::chrono::milliseconds(start / sampleRateMs_);
        auto endDuration = std::chrono::milliseconds(end / sampleRateMs_);
        return fmt::format("ms: {:>8} -> {:>8}", startDuration.count(), endDuration.count());
        // return fmt::format("timestamp {:08d}, {:08d}", start, end);
        // return format("{start:%08d,end:%08d}", start, end);
    };
private:
    int sampleRateMs_;
};


class VadIterator {
private:
    // OnnxRuntime resources
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::shared_ptr<Ort::Session> session_{};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memoryInfo_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

private:
    void init_engine_threads(int interThreads, int intraThreads) {
        // The method should be called in each thread/proc in multi-thread/proc work
        sessionOptions_.SetIntraOpNumThreads(intraThreads);
        sessionOptions_.SetInterOpNumThreads(interThreads);
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void init_onnx_model(const std::string &modelPath) {
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session_ = std::make_shared<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
    };

    void reset_states() {
        // Call reset before each audio start
        std::memset(h_.data(), 0.0f, h_.size() * sizeof(float));
        std::memset(c_.data(), 0.0f, c_.size() * sizeof(float));
        triggered_ = false;
        tempEnd_ = 0;
        sampleIndex_ = 0;

        prevEnd_ = nextStart_ = 0;

        speeches_.clear();
        currentSpeech_ = timestamp_t(sampleRateMs_);
        state_ = speechState::Off;
        current_ = 0;
        up_ = 0;
        down_ = 0;
    };

    void predict(const std::vector<float> &data, uint index) {
        // Infer
        // Create ort tensors
        input_.assign(data.begin(), data.end());
        Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
                memoryInfo_, input_.data(), input_.size(), inputNodeDims_, 2);
        Ort::Value srOrt = Ort::Value::CreateTensor<int64_t>(
                memoryInfo_, sr_.data(), sr_.size(), srNodeDims_, 1);
        Ort::Value hOrt = Ort::Value::CreateTensor<float>(
                memoryInfo_, h_.data(), h_.size(), hcNodeDims_, 3);
        Ort::Value cOrt = Ort::Value::CreateTensor<float>(
                memoryInfo_, c_.data(), c_.size(), hcNodeDims_, 3);

        // Clear and add inputs
        ortInputs_.clear();
        ortInputs_.emplace_back(std::move(inputOrt));
        ortInputs_.emplace_back(std::move(srOrt));
        ortInputs_.emplace_back(std::move(hOrt));
        ortInputs_.emplace_back(std::move(cOrt));

        // Infer
        ortOutputs_ = session_->Run(
                Ort::RunOptions{nullptr},
                inputNodeNames_.data(), ortInputs_.data(), ortInputs_.size(),
                outputNodeNames_.data(), outputNodeNames_.size());

        // Output probability & update h,c recursively
        float speechProb = ortOutputs_[0].GetTensorMutableData<float>()[0];
        auto *hn = ortOutputs_[1].GetTensorMutableData<float>();
        std::memcpy(h_.data(), hn, hcSize_ * sizeof(float));
        auto *cn = ortOutputs_[2].GetTensorMutableData<float>();
        std::memcpy(c_.data(), cn, hcSize_ * sizeof(float));


        // current_ += subSamples_;
        current_ = index;
        switch (state_) {
            case speechState::Off:
                if (speechProb >= thresholdHigh_) {
                    up_ = current_ - samples_;
                    state_ = speechState::MaybeOn;
                }
                break;
            case speechState::MaybeOn:
                if (speechProb >= thresholdHigh_) {
                    if (current_ - up_ >= minSpeechSamples_) {
                        currentSpeech_.start = int(up_);
                        state_ = speechState::On;
                    }
                }
                if (speechProb < thresholdLow_) {
                    state_ = speechState::Off;
                }
                break;
            case speechState::On:
                if (speechProb < thresholdLow_) {
                    down_ = current_ - samples_;
                    state_ = speechState::MaybeOff;
                }
                break;
            case speechState::MaybeOff:
                if (speechProb >= thresholdHigh_) {
                    state_ = speechState::On;
                }
                if (speechProb < thresholdLow_) {
                    if (current_ - down_ >= minSilenceSamples_) {
                        currentSpeech_.end = int(down_);
                        state_ = speechState::Off;

                        speeches_.push_back(currentSpeech_);
                        currentSpeech_ = timestamp_t(sampleRateMs_);
                        up_ = 0;
                        down_ = 0;
                    }
                }
                break;
        }

//         // Push forward sample index
//         sampleIndex_ += samples_;
//
//         // Reset temp_end when > threshold
//         if (speechProb >= threshold_) {
// #ifdef __DEBUG_SPEECH_PROB___
//             float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
//             printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample- window_size_samples);
// #endif //__DEBUG_SPEECH_PROB___
//             if (tempEnd_ != 0) {
//                 tempEnd_ = 0;
//                 // USELESS
//                 // if (nextStart_ < prevEnd_)
//                 //     nextStart_ = sampleIndex_ - samples_;
//             }
//             if (!triggered_) {
//                 triggered_ = true;
//                 currentSpeech_.start = sampleIndex_ - samples_;
//             }
//         }
//
//         // USELESS
//         // if (triggered_ && (float(sampleIndex_ - currentSpeech_.start) > maxSpeechSamples_)) {
//         //     if (prevEnd_ > 0) {
//         //         currentSpeech_.end = prevEnd_;
//         //         speeches_.push_back(currentSpeech_);
//         //         currentSpeech_ = timestamp_t(sampleRateMs_);
//         //
//         //         // previously reached silence(< neg_thres) and is still not speech(< thres)
//         //         if (nextStart_ < prevEnd_)
//         //             triggered_ = false;
//         //         else
//         //             currentSpeech_.start = nextStart_;
//         //
//         //         prevEnd_ = 0;
//         //         nextStart_ = 0;
//         //         tempEnd_ = 0;
//         //
//         //     } else {
//         //         currentSpeech_.end = sampleIndex_;
//         //         speeches_.push_back(currentSpeech_);
//         //         currentSpeech_ = timestamp_t(sampleRateMs_);
//         //         prevEnd_ = 0;
//         //         nextStart_ = 0;
//         //         tempEnd_ = 0;
//         //         triggered_ = false;
//         //     }
//         //     return;
//         //
//         // }
//
//         else if (speechProb >= (threshold_ - 0.15)) {
//             if (triggered_) {
// #ifdef __DEBUG_SPEECH_PROB___
//                 float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
//                 printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
//             } else {
//                 float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
//                 printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
// #endif //__DEBUG_SPEECH_PROB___
//             }
//         }
//
//
//         // 4) End
//         // if ((speechProb < (threshold_ - 0.15))) {
//         else {
// #ifdef __DEBUG_SPEECH_PROB___
//             float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
//             printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
// #endif //__DEBUG_SPEECH_PROB___
//             if (triggered_) {
//                 if (tempEnd_ == 0) {
//                     tempEnd_ = sampleIndex_;
//                 }
//                 // USELESS
//                 // if (sampleIndex_ - tempEnd_ > minSilenceSamplesAtMaxSpeech_)
//                 //     prevEnd_ = tempEnd_;
//                 // a. silence < min_slience_samples, continue speaking
//                 if ((sampleIndex_ - tempEnd_) < minSilenceSamples_) {
//
//                 }
//                     // b. silence >= min_slience_samples, end speaking
//                 else {
//                     currentSpeech_.end = tempEnd_;
//                     if (currentSpeech_.end - currentSpeech_.start > minSpeechSamples_) {
//                         speeches_.push_back(currentSpeech_);
//                         currentSpeech_ = timestamp_t(sampleRateMs_);
//                         prevEnd_ = 0;
//                         nextStart_ = 0;
//                         tempEnd_ = 0;
//                         triggered_ = false;
//                     }
//                 }
//             } else {
//                 // may first windows see end state.
//             }
//             return;
//         }
    };
public:
    void process(const std::vector<float> &input_wav) {
        reset_states();

        audioLengthSamples_ = input_wav.size();

        // for (int j = 0; j < audioLengthSamples_; j += samples_) {
        //     if (j + samples_ > audioLengthSamples_)
        //         break;
        //     std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + samples_};
        //     predict(r);
        // }
        for (uint index = subSamples_; index < audioLengthSamples_; index += subSamples_) {
            if (index + subSamples_ > audioLengthSamples_)
                break;
            std::vector<float> r{&input_wav[0] + index - subSamples_, &input_wav[0] + index + subSamples_};
            predict(r, index + subSamples_);
        }

        if (currentSpeech_.start >= 0) {
            currentSpeech_.end = audioLengthSamples_;
            speeches_.push_back(currentSpeech_);
            currentSpeech_ = timestamp_t(sampleRateMs_);
            prevEnd_ = 0;
            nextStart_ = 0;
            tempEnd_ = 0;
            triggered_ = false;
        }
    };

    void process(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
        process(input_wav);
        collect_chunks(input_wav, output_wav);
    }

    void collect_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
        output_wav.clear();
        for (auto & speech : speeches_) {
#ifdef __DEBUG_SPEECH_PROB___
            std::cout << speeches[i].c_str() << std::endl;
#endif //#ifdef __DEBUG_SPEECH_PROB___
            std::vector<float> slice(&input_wav[speech.start], &input_wav[speech.end]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
        }
    };

    const std::vector<timestamp_t> get_speech_timestamps() const {
        return speeches_;
    }

    void drop_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
        output_wav.clear();
        int current_start = 0;
        for (auto & speech : speeches_) {

            std::vector<float> slice(&input_wav[current_start], &input_wav[speech.start]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
            current_start = speech.end;
        }

        std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    };

private:
    // model config
    unsigned int samples_{};  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    unsigned int subSamples_{};
    int sampleRate_{};  //Assign when init support 16000 or 8000
    int sampleRateMs_;   // Assign when init, support 8 or 16
    float thresholdHigh_{};
    float thresholdLow_{};
    int minSilenceSamples_{};
    std::chrono::milliseconds minSilenceDuration_{}; // sr_per_ms * #ms
    int minSilenceSamplesAtMaxSpeech_; // sr_per_ms * #98
    int minSpeechSamples_{}; // sr_per_ms * #ms
    float maxSpeechSamples_;
    int padSpeechSamples_{}; // usually a
    int audioLengthSamples_{};

    // model states
    bool triggered_ = false;
    unsigned int tempEnd_{};
    unsigned int sampleIndex_{};

    unsigned int current_{};
    unsigned int up_{};
    unsigned int down_{};
    speechState state_{Off};



    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
    int prevEnd_{};
    int nextStart_{};

    //Output timestamp
    std::vector<timestamp_t> speeches_;
    timestamp_t currentSpeech_{sampleRateMs_};


    // Onnx model
    // Inputs
    std::vector<Ort::Value> ortInputs_;

    std::vector<const char *> inputNodeNames_ = {"input", "sr", "h", "c"};
    std::vector<float> input_;
    std::vector<int64_t> sr_;
    unsigned int hcSize_ = 2 * 1 * 64; // It's FIXED.
    std::vector<float> h_;
    std::vector<float> c_;

    int64_t inputNodeDims_[2] = {};
    const int64_t srNodeDims_[1] = {1};
    const int64_t hcNodeDims_[3] = {2, 1, 64};

    // Outputs
    std::vector<Ort::Value> ortOutputs_;
    std::vector<const char *> outputNodeNames_ = {"output", "hn", "cn"};

public:
    // Construction
    explicit VadIterator(const std::string& modelPath,
                         int sampleRate = 16000, std::chrono::milliseconds samplingDuration = 64ms,
                         float threshold = 0.8, std::chrono::milliseconds minSilenceDuration = 256ms,
                         std::chrono::milliseconds speechPadDuration = 64ms, std::chrono::milliseconds minSpeechDuration = 256ms,
                         float max_speech_duration_s = std::numeric_limits<float>::infinity()) : thresholdHigh_{threshold}, thresholdLow_{threshold - 0.15f}, sampleRate_{sampleRate} {
        init_onnx_model(modelPath);
        sampleRateMs_ = sampleRate / 1000;

        samples_ = samplingDuration.count() * sampleRateMs_;
        subSamples_ = samples_ / 2;

        minSpeechSamples_ = minSpeechDuration.count() * sampleRateMs_;
        padSpeechSamples_ = speechPadDuration.count() * sampleRateMs_;

        maxSpeechSamples_ = (
                sampleRate * max_speech_duration_s
                - samples_
                - 2 * padSpeechSamples_
        );

        minSilenceSamples_ = minSilenceDuration.count() * sampleRateMs_;
        minSilenceSamplesAtMaxSpeech_ = sampleRateMs_ * 98;

        input_.resize(samples_);
        inputNodeDims_[0] = 1;
        inputNodeDims_[1] = samples_;

        h_.resize(hcSize_);
        c_.resize(hcSize_);
        sr_.resize(1);
        sr_[0] = sampleRate;
    };
};

int main() {
    std::vector<timestamp_t> stamps;

    // Read wav
    wav::WavReader wav_reader("recorder2.wav"); //16000,1,32float
    std::vector<float> input_wav(wav_reader.num_samples());
    std::vector<float> output_wav;

    for (int i = 0; i < wav_reader.num_samples(); i++) {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }



    // ===== Test configs =====
    std::string path = "silero_vad.onnx";
    VadIterator vad(path);

    // ==============================================
    // ==== = Example 1 of full function  ===== 
    // ==============================================
    vad.process(input_wav);

    // 1.a get_speech_timestamps
    stamps = vad.get_speech_timestamps();
    for (auto &stamp: stamps)
        std::cout << stamp.c_str() << std::endl;

    // // 1.b collect_chunks output wav
    // vad.collect_chunks(input_wav, output_wav);
    //
    // // 1.c drop_chunks output wav
    // vad.drop_chunks(input_wav, output_wav);
    //
    // // ==============================================
    // // ===== Example 2 of simple full function  =====
    // // ==============================================
    // vad.process(input_wav, output_wav);
    //
    // stamps = vad.get_speech_timestamps();
    // for (auto &stamp: stamps) {
    //
    //     std::cout << stamp.c_str() << std::endl;
    // }
    //
    // // ==============================================
    // // ===== Example 3 of full function  =====
    // // ==============================================
    // for (int i = 0; i < 2; i++)
    //     vad.process(input_wav, output_wav);
}
