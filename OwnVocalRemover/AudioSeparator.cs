using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Ownaudio;
using Ownaudio.Engines;
using Ownaudio.Sources;


namespace OwnSeparator.Core
{
    /// <summary>
    /// Configuration parameters for audio separation process
    /// </summary>
    public class SeparationOptions
    {
        /// <summary>
        /// ONNX model file path
        /// </summary>
        public string ModelPath { get; set; } = "models/MODERN_INST_DEFAULT.onnx";

        /// <summary>
        /// Output directory path
        /// </summary>
        public string OutputDirectory { get; set; } = "separated";

        /// <summary>
        /// Disable noise reduction (enabled by default)
        /// </summary>
        public bool DisableNoiseReduction { get; set; } = false;

        /// <summary>
        /// Margin size for overlapping chunks (in samples)
        /// </summary>
        public int Margin { get; set; } = 44100;

        /// <summary>
        /// Chunk size in seconds (0 = process entire file at once)
        /// </summary>
        public int ChunkSizeSeconds { get; set; } = 15;

        /// <summary>
        /// FFT size
        /// </summary>
        public int NFft { get; set; } = 6144;

        /// <summary>
        /// Temporal dimension parameter (as power of 2)
        /// </summary>
        public int DimT { get; set; } = 8;

        /// <summary>
        /// Frequency dimension parameter
        /// </summary>
        public int DimF { get; set; } = 2048;
    }

    /// <summary>
    /// Progress information for separation process
    /// </summary>
    public class SeparationProgress
    {
        /// <summary>
        /// Current file being processed
        /// </summary>
        public string CurrentFile { get; set; } = string.Empty;

        /// <summary>
        /// Overall progress percentage (0-100)
        /// </summary>
        public double OverallProgress { get; set; }

        /// <summary>
        /// Current chunk progress percentage (0-100)
        /// </summary>
        public double ChunkProgress { get; set; }

        /// <summary>
        /// Current processing step description
        /// </summary>
        public string Status { get; set; } = string.Empty;

        /// <summary>
        /// Number of chunks processed
        /// </summary>
        public int ProcessedChunks { get; set; }

        /// <summary>
        /// Total number of chunks
        /// </summary>
        public int TotalChunks { get; set; }
    }

    /// <summary>
    /// Result of audio separation
    /// </summary>
    public class SeparationResult
    {
        /// <summary>
        /// Path to the vocals output file
        /// </summary>
        public string VocalsPath { get; set; } = string.Empty;

        /// <summary>
        /// Path to the instrumental output file
        /// </summary>
        public string InstrumentalPath { get; set; } = string.Empty;

        /// <summary>
        /// Processing duration
        /// </summary>
        public TimeSpan ProcessingTime { get; set; }

        /// <summary>
        /// Audio statistics
        /// </summary>
        public AudioStatistics Statistics { get; set; } = new AudioStatistics();
    }

    /// <summary>
    /// Audio processing statistics
    /// </summary>
    public class AudioStatistics
    {
        public double MixRMS { get; set; }
        public double VocalsRMS { get; set; }
        public double InstrumentalRMS { get; set; }
        public double VocalsMixRatio { get; set; }
        public double InstrumentalMixRatio { get; set; }
        public int SampleRate { get; set; }
        public int Channels { get; set; }
        public int SampleCount { get; set; }
    }

    /// <summary>
    /// Main audio separation service
    /// </summary>
    public class AudioSeparationService : IDisposable
    {
        #region Events

        /// <summary>
        /// Progress update event
        /// </summary>
        public event EventHandler<SeparationProgress>? ProgressChanged;

        /// <summary>
        /// Processing started event
        /// </summary>
        public event EventHandler<string>? ProcessingStarted;

        /// <summary>
        /// Processing completed event
        /// </summary>
        public event EventHandler<SeparationResult>? ProcessingCompleted;

        /// <summary>
        /// Error occurred event
        /// </summary>
        public event EventHandler<Exception>? ErrorOccurred;

        #endregion

        #region Private Fields

        private readonly SeparationOptions _options;
        private ModelParameters _modelParams;
        private InferenceSession? _onnxSession;
        private bool _disposed = false;
        private const int TargetSampleRate = 44100;

        #endregion

        #region Constructor

        /// <summary>
        /// Initialize audio separation service
        /// </summary>
        /// <param name="options">Separation configuration options</param>
        public AudioSeparationService(SeparationOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _modelParams = new ModelParameters(
                dimF: options.DimF,
                dimT: options.DimT,
                nFft: options.NFft
            );
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the ONNX model session and auto-detect model dimensions
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task InitializeAsync(CancellationToken cancellationToken = default)
        {
            await Task.Run(() =>
            {
                if (!File.Exists(_options.ModelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {_options.ModelPath}");
                }

                var sessionOptions = new SessionOptions
                {
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };

                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA();
                    Console.WriteLine("CUDA execution provider enabled.");
                }
                catch
                {
                    sessionOptions.AppendExecutionProvider_CPU();
                    Console.WriteLine("Using CPU execution provider.");
                }

                _onnxSession = new InferenceSession(_options.ModelPath, sessionOptions);

                // Auto-detect model dimensions
                AutoDetectModelDimensions();

                Console.WriteLine($"Model parameters: DimF={_modelParams.DimF}, DimT={_modelParams.DimT}, NFft={_modelParams.NFft}");
            }, cancellationToken);
        }

        /// <summary>
        /// Separate audio file into vocals and instrumental tracks
        /// </summary>
        /// <param name="inputFilePath">Input audio file path</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Separation result</returns>
        public async Task<SeparationResult> SeparateAsync(string inputFilePath, CancellationToken cancellationToken = default)
        {
            if (_onnxSession == null)
                throw new InvalidOperationException("Service not initialized. Call InitializeAsync first.");

            if (!File.Exists(inputFilePath))
                throw new FileNotFoundException($"Input file not found: {inputFilePath}");

            var startTime = DateTime.Now;
            var filename = Path.GetFileNameWithoutExtension(inputFilePath);

            try
            {
                ProcessingStarted?.Invoke(this, inputFilePath);

                ReportProgress(new SeparationProgress
                {
                    CurrentFile = Path.GetFileName(inputFilePath),
                    Status = "Loading audio file...",
                    OverallProgress = 0
                });

                var mix = await LoadAndPrepareAudioAsync(inputFilePath, cancellationToken);

                ReportProgress(new SeparationProgress
                {
                    CurrentFile = Path.GetFileName(inputFilePath),
                    Status = "Processing audio separation...",
                    OverallProgress = 10
                });

                var separated = await ProcessAudioAsync(mix, cancellationToken);

                ReportProgress(new SeparationProgress
                {
                    CurrentFile = Path.GetFileName(inputFilePath),
                    Status = "Calculating results...",
                    OverallProgress = 90
                });

                // Calculate vocals and instrumental
                var vocals = new float[2, mix.GetLength(1)];
                var instrumental = new float[2, mix.GetLength(1)];

                for (int ch = 0; ch < 2; ch++)
                {
                    for (int i = 0; i < mix.GetLength(1); i++)
                    {
                        vocals[ch, i] = mix[ch, i] - separated[ch, i];      // mix - separated
                        instrumental[ch, i] = separated[ch, i];             // separated
                    }
                }

                var statistics = CalculateStatistics(mix, vocals, instrumental);

                Directory.CreateDirectory(_options.OutputDirectory);

                var modelName = Path.GetFileName(_options.ModelPath).ToUpper();
                var (vocalsPath, instrumentalPath) = await SaveResultsAsync(
                    filename, vocals, instrumental, statistics.SampleRate, modelName, cancellationToken);

                var result = new SeparationResult
                {
                    VocalsPath = vocalsPath,
                    InstrumentalPath = instrumentalPath,
                    ProcessingTime = DateTime.Now - startTime,
                    Statistics = statistics
                };

                ReportProgress(new SeparationProgress
                {
                    CurrentFile = Path.GetFileName(inputFilePath),
                    Status = "Completed",
                    OverallProgress = 100
                });

                ProcessingCompleted?.Invoke(this, result);
                return result;
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(this, ex);
                throw;
            }
        }

        /// <summary>
        /// Separate multiple audio files
        /// </summary>
        /// <param name="inputFilePaths">Input audio file paths</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>List of separation results</returns>
        public async Task<List<SeparationResult>> SeparateMultipleAsync(
            IEnumerable<string> inputFilePaths,
            CancellationToken cancellationToken = default)
        {
            var results = new List<SeparationResult>();
            var files = inputFilePaths.ToList();

            for (int i = 0; i < files.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var result = await SeparateAsync(files[i], cancellationToken);
                results.Add(result);

                var overallProgress = (double)(i + 1) / files.Count * 100;
                ReportProgress(new SeparationProgress
                {
                    CurrentFile = "Batch processing",
                    Status = $"Completed {i + 1} of {files.Count} files",
                    OverallProgress = overallProgress
                });
            }

            return results;
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _onnxSession?.Dispose();
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        private void AutoDetectModelDimensions()
        {
            if (_onnxSession == null) return;

            try
            {
                var inputMetadata = _onnxSession.InputMetadata;
                if (inputMetadata.ContainsKey("input"))
                {
                    var inputShape = inputMetadata["input"].Dimensions;

                    if (inputShape.Length >= 4)
                    {
                        // Expected shape: [batch, channels, frequency, time]
                        int expectedFreq = (int)inputShape[2];
                        int expectedTime = (int)inputShape[3];

                        Console.WriteLine($"Model expects: Frequency={expectedFreq}, Time={expectedTime}");
                        Console.WriteLine($"Current config: Frequency={_modelParams.DimF}, Time={_modelParams.DimT}");

                        // Update model parameters if they don't match
                        if (expectedFreq != _modelParams.DimF || expectedTime != _modelParams.DimT)
                        {
                            Console.WriteLine("Auto-adjusting model parameters to match ONNX model...");

                            int newDimT = (int)Math.Log2(expectedTime);

                            _modelParams = new ModelParameters(
                                dimF: expectedFreq,
                                dimT: newDimT,
                                nFft: _options.NFft
                            );

                            Console.WriteLine($"Updated to: DimF={_modelParams.DimF}, DimT={_modelParams.DimT}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not auto-detect model dimensions: {ex.Message}");
                Console.WriteLine("Using provided configuration parameters...");
            }
        }

        private void ReportProgress(SeparationProgress progress)
        {
            ProgressChanged?.Invoke(this, progress);
        }

        private async Task<float[,]> LoadAndPrepareAudioAsync(string filePath, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                OwnAudio.Initialize();

                AudioEngineOutputOptions _audioEngineOptions = new AudioEngineOutputOptions
                (
                    device: OwnAudio.DefaultOutputDevice,
                    channels: OwnAudioEngine.EngineChannels.Stereo,
                    sampleRate: 44100,
                    latency: OwnAudio.DefaultOutputDevice.DefaultHighOutputLatency
                );

                SourceManager.OutputEngineOptions = _audioEngineOptions;
                SourceManager _manager = SourceManager.Instance;

                _manager.AddOutputSource( filePath );

                List<float> samples = _manager.Sources[0].GetFloatAudioData(new TimeSpan(0)).ToList();

                int channels = (int)_audioEngineOptions.Channels;
                int frameCount = samples.Count / channels;
                var audioBuffer = new float[channels, frameCount];

                for (int i = 0; i < frameCount; i++)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        audioBuffer[ch, i] = samples[i * channels + ch];
                    }
                }

                OwnAudio.Free();

                return audioBuffer;
            }, cancellationToken);
        }

        private async Task<float[,]> ProcessAudioAsync(float[,] mix, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                int samples = mix.GetLength(1);
                int margin = _options.Margin;
                int chunkSize = _options.ChunkSizeSeconds * TargetSampleRate;

                if (margin == 0) throw new ArgumentException("Margin cannot be zero!");
                if (chunkSize != 0 && margin > chunkSize) margin = chunkSize;
                if (_options.ChunkSizeSeconds == 0 || samples < chunkSize) chunkSize = samples;

                var chunks = CreateChunks(mix, chunkSize, margin);
                return ProcessChunks(chunks, margin, cancellationToken);
            }, cancellationToken);
        }

        private Dictionary<long, float[,]> CreateChunks(float[,] mix, int chunkSize, int margin)
        {
            var chunks = new Dictionary<long, float[,]>();
            int samples = mix.GetLength(1);
            long counter = -1;

            for (long skip = 0; skip < samples; skip += chunkSize)
            {
                counter++;
                long sMargin = counter == 0 ? 0 : margin;
                long end = Math.Min(skip + chunkSize + margin, samples);
                long start = skip - sMargin;
                int segmentLength = (int)(end - start);

                var segment = new float[2, segmentLength];
                for (int ch = 0; ch < 2; ch++)
                {
                    for (int i = 0; i < segmentLength; i++)
                    {
                        segment[ch, i] = mix[ch, start + i];
                    }
                }
                chunks[skip] = segment;
                if (end == samples) break;
            }

            return chunks;
        }

        private float[,] ProcessChunks(Dictionary<long, float[,]> chunks, int margin, CancellationToken cancellationToken)
        {
            var processedChunks = new List<float[,]>();
            var keys = chunks.Keys.ToList();
            int totalChunks = chunks.Count;

            ReportProgress(new SeparationProgress
            {
                Status = "Processing chunks...",
                TotalChunks = totalChunks,
                ProcessedChunks = 0,
                OverallProgress = 20
            });

            int processedCount = 0;
            foreach (var kvp in chunks)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var chunk = ProcessSingleChunk(kvp.Value, kvp.Key, keys, margin);
                processedChunks.Add(chunk);

                processedCount++;
                double chunkProgress = (double)processedCount / totalChunks * 100;

                ReportProgress(new SeparationProgress
                {
                    Status = $"Processing chunk {processedCount}/{totalChunks}",
                    ChunkProgress = chunkProgress,
                    ProcessedChunks = processedCount,
                    TotalChunks = totalChunks,
                    OverallProgress = 20 + (chunkProgress * 0.6)
                });
            }

            return ConcatenateChunks(processedChunks);
        }

        private float[,] ProcessSingleChunk(float[,] mixChunk, long chunkKey, List<long> allKeys, int margin)
        {
            int nSample = mixChunk.GetLength(1);
            int trim = _modelParams.NFft / 2;
            int genSize = _modelParams.ChunkSize - 2 * trim;

            if (genSize <= 0)
                throw new ArgumentException($"Invalid genSize: {genSize}. Check FFT parameters.");

            // Padding
            int pad = genSize - (nSample % genSize);
            if (nSample % genSize == 0) pad = 0;

            var mixPadded = new float[2, trim + nSample + pad + trim];
            for (int ch = 0; ch < 2; ch++)
            {
                for (int i = 0; i < nSample; i++)
                {
                    mixPadded[ch, trim + i] = mixChunk[ch, i];
                }
            }

            int frameCount = (nSample + pad) / genSize;
            var mixWaves = new float[frameCount, 2, _modelParams.ChunkSize];

            for (int i = 0; i < frameCount; i++)
            {
                int offset = i * genSize;
                for (int ch = 0; ch < 2; ch++)
                {
                    for (int j = 0; j < _modelParams.ChunkSize; j++)
                    {
                        mixWaves[i, ch, j] = mixPadded[ch, offset + j];
                    }
                }
            }

            // STFT -> Model -> ISTFT
            var stftTensor = ComputeStft(mixWaves);
            var outputTensor = RunModelInference(stftTensor);
            var resultWaves = ComputeIstft(outputTensor);

            // Extract and apply margin
            var result = ExtractSignal(resultWaves, nSample, trim, genSize);
            return ApplyMargin(result, chunkKey, allKeys, margin);
        }

        private DenseTensor<float> ComputeStft(float[,,] mixWaves)
        {
            int batchSize = mixWaves.GetLength(0);
            var tensor = new DenseTensor<float>(new[] { batchSize, 4, _modelParams.DimF, _modelParams.DimT });

            for (int b = 0; b < batchSize; b++)
            {
                for (int ch = 0; ch < 2; ch++)
                {
                    int padSize = _modelParams.NFft / 2;
                    var paddedSignal = new float[_modelParams.ChunkSize + 2 * padSize];

                    // Reflection padding
                    for (int i = 0; i < padSize; i++)
                    {
                        int srcIdx = Math.Min(padSize - 1 - i, _modelParams.ChunkSize - 1);
                        paddedSignal[i] = mixWaves[b, ch, srcIdx];
                    }

                    for (int i = 0; i < _modelParams.ChunkSize; i++)
                    {
                        paddedSignal[padSize + i] = mixWaves[b, ch, i];
                    }

                    for (int i = 0; i < padSize; i++)
                    {
                        int srcIdx = Math.Max(0, _modelParams.ChunkSize - 1 - i);
                        paddedSignal[padSize + _modelParams.ChunkSize + i] = mixWaves[b, ch, srcIdx];
                    }

                    // STFT computation
                    for (int t = 0; t < _modelParams.DimT; t++)
                    {
                        int frameStart = t * _modelParams.Hop;
                        var frame = new Complex[_modelParams.NFft];

                        for (int i = 0; i < _modelParams.NFft; i++)
                        {
                            if (frameStart + i < paddedSignal.Length)
                            {
                                double windowValue = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / _modelParams.NFft));
                                frame[i] = new Complex(paddedSignal[frameStart + i] * windowValue, 0);
                            }
                        }

                        Fourier.Forward(frame, FourierOptions.NoScaling);

                        for (int f = 0; f < Math.Min(_modelParams.DimF, _modelParams.NBins); f++)
                        {
                            tensor[b, ch * 2, f, t] = (float)frame[f].Real;
                            tensor[b, ch * 2 + 1, f, t] = (float)frame[f].Imaginary;
                        }
                    }
                }
            }
            return tensor;
        }

        private Tensor<float> RunModelInference(DenseTensor<float> stftTensor)
        {
            if (_onnxSession == null)
                throw new InvalidOperationException("ONNX session not initialized");

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", stftTensor) };

            if (!_options.DisableNoiseReduction)
            {
                // Denoise logic
                var stftTensorNeg = new DenseTensor<float>(stftTensor.Dimensions);
                for (int idx = 0; idx < stftTensor.Length; idx++)
                {
                    stftTensorNeg.SetValue(idx, -stftTensor.GetValue(idx));
                }
                var inputsNeg = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", stftTensorNeg) };

                using var outputs = _onnxSession.Run(inputs);
                using var outputsNeg = _onnxSession.Run(inputsNeg);

                var specPred = outputs.First().AsTensor<float>();
                var specPredNeg = outputsNeg.First().AsTensor<float>();

                var result = new DenseTensor<float>(specPred.Dimensions);

                for (int b = 0; b < specPred.Dimensions[0]; b++)
                    for (int c = 0; c < specPred.Dimensions[1]; c++)
                        for (int f = 0; f < specPred.Dimensions[2]; f++)
                            for (int t = 0; t < specPred.Dimensions[3]; t++)
                            {
                                float val = -specPredNeg[b, c, f, t] * 0.5f + specPred[b, c, f, t] * 0.5f;
                                ((DenseTensor<float>)result)[b, c, f, t] = val;
                            }

                return result;
            }
            else
            {
                using var outputs = _onnxSession.Run(inputs);
                var result = outputs.First().AsTensor<float>();

                // Create a copy to avoid disposal issues
                var resultCopy = new DenseTensor<float>(result.Dimensions);
                for (int i = 0; i < result.Length; i++)
                {
                    resultCopy.SetValue(i, result.GetValue(i));
                }
                return resultCopy;
            }
        }

        private float[,,] ComputeIstft(Tensor<float> spectrum)
        {
            int batchSize = spectrum.Dimensions[0];
            var result = new float[batchSize, 2, _modelParams.ChunkSize];

            for (int b = 0; b < batchSize; b++)
            {
                for (int ch = 0; ch < 2; ch++)
                {
                    int padSize = _modelParams.NFft / 2;
                    var reconstructed = new double[_modelParams.ChunkSize + 2 * padSize];
                    var windowSum = new double[_modelParams.ChunkSize + 2 * padSize];

                    int realIdx = ch * 2;
                    int imagIdx = ch * 2 + 1;

                    for (int t = 0; t < _modelParams.DimT; t++)
                    {
                        var frame = new Complex[_modelParams.NFft];

                        for (int f = 0; f < _modelParams.NBins && f < _modelParams.NFft; f++)
                        {
                            if (f < _modelParams.DimF && f < spectrum.Dimensions[2])
                            {
                                frame[f] = new Complex(spectrum[b, realIdx, f, t], spectrum[b, imagIdx, f, t]);
                            }
                            else
                            {
                                frame[f] = Complex.Zero;
                            }
                        }

                        // Hermitian symmetry
                        for (int f = 1; f < _modelParams.NFft / 2; f++)
                        {
                            if (_modelParams.NFft - f < frame.Length)
                            {
                                frame[_modelParams.NFft - f] = Complex.Conjugate(frame[f]);
                            }
                        }

                        Fourier.Inverse(frame, FourierOptions.NoScaling);

                        for (int i = 0; i < _modelParams.NFft; i++)
                        {
                            frame[i] /= _modelParams.NFft;
                        }

                        int frameStart = t * _modelParams.Hop;
                        for (int i = 0; i < _modelParams.NFft; i++)
                        {
                            int targetIdx = frameStart + i;
                            if (targetIdx >= 0 && targetIdx < reconstructed.Length)
                            {
                                double windowValue = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / _modelParams.NFft));
                                reconstructed[targetIdx] += frame[i].Real * windowValue;
                                windowSum[targetIdx] += windowValue * windowValue;
                            }
                        }
                    }

                    for (int i = 0; i < _modelParams.ChunkSize; i++)
                    {
                        int srcIdx = i + padSize;
                        if (srcIdx >= 0 && srcIdx < reconstructed.Length)
                        {
                            if (windowSum[srcIdx] > 1e-10)
                            {
                                result[b, ch, i] = (float)(reconstructed[srcIdx] / windowSum[srcIdx]);
                            }
                            else
                            {
                                result[b, ch, i] = (float)reconstructed[srcIdx];
                            }
                        }
                    }
                }
            }
            return result;
        }

        private float[,] ExtractSignal(float[,,] waves, int nSample, int trim, int genSize)
        {
            int frameCount = waves.GetLength(0);
            var signal = new float[2, nSample];

            for (int i = 0; i < frameCount; i++)
            {
                int destOffset = i * genSize;
                for (int ch = 0; ch < 2; ch++)
                {
                    for (int j = 0; j < genSize && destOffset + j < nSample; j++)
                    {
                        int sourceIndex = trim + j;
                        if (sourceIndex < _modelParams.ChunkSize - trim)
                        {
                            signal[ch, destOffset + j] = waves[i, ch, sourceIndex];
                        }
                    }
                }
            }

            return signal;
        }

        private float[,] ApplyMargin(float[,] signal, long chunkKey, List<long> allKeys, int margin)
        {
            int nSample = signal.GetLength(1);
            int start = chunkKey == 0 ? 0 : margin;
            int end = chunkKey == allKeys.Last() ? nSample : nSample - margin;
            if (margin == 0) end = nSample;

            var result = new float[2, end - start];
            for (int ch = 0; ch < 2; ch++)
            {
                for (int i = 0; i < end - start; i++)
                {
                    result[ch, i] = signal[ch, start + i];
                }
            }

            return result;
        }

        private float[,] ConcatenateChunks(List<float[,]> chunks)
        {
            int totalLength = chunks.Sum(c => c.GetLength(1));
            var result = new float[2, totalLength];
            int currentPos = 0;

            foreach (var chunk in chunks)
            {
                for (int ch = 0; ch < 2; ch++)
                {
                    for (int i = 0; i < chunk.GetLength(1); i++)
                    {
                        result[ch, currentPos + i] = chunk[ch, i];
                    }
                }
                currentPos += chunk.GetLength(1);
            }

            return result;
        }

        private AudioStatistics CalculateStatistics(float[,] mix, float[,] vocals, float[,] instrumental)
        {
            double mixRMS = CalculateRMS(mix);
            double vocalsRMS = CalculateRMS(vocals);
            double instrumentalRMS = CalculateRMS(instrumental);

            return new AudioStatistics
            {
                MixRMS = mixRMS,
                VocalsRMS = vocalsRMS,
                InstrumentalRMS = instrumentalRMS,
                VocalsMixRatio = vocalsRMS / mixRMS,
                InstrumentalMixRatio = instrumentalRMS / mixRMS,
                SampleRate = TargetSampleRate,
                Channels = mix.GetLength(0),
                SampleCount = mix.GetLength(1)
            };
        }

        private double CalculateRMS(float[,] audio)
        {
            double sum = 0;
            int channels = audio.GetLength(0);
            int samples = audio.GetLength(1);

            for (int ch = 0; ch < channels; ch++)
            {
                for (int i = 0; i < samples; i++)
                {
                    sum += audio[ch, i] * audio[ch, i];
                }
            }

            return Math.Sqrt(sum / (channels * samples));
        }

        private async Task<(string vocalsPath, string instrumentalPath)> SaveResultsAsync(
            string filename, float[,] vocals, float[,] instrumental, int sampleRate, string modelName, CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                var vocalsPath = Path.Combine(_options.OutputDirectory, $"{filename}_vocals.wav");
                var instrumentalPath = Path.Combine(_options.OutputDirectory, $"{filename}_music.wav");

                // Python model-specific logic reproduction
                if (modelName != "OWN_INST_DEFAULT.ONNX")
                {
                    SaveAudio(instrumentalPath, instrumental, sampleRate);
                    SaveAudio(vocalsPath, vocals, sampleRate);
                }
                else
                {
                    // OWN_INST_DEFAULT: swap assignment
                    SaveAudio(instrumentalPath, vocals, sampleRate);
                    SaveAudio(vocalsPath, instrumental, sampleRate);
                }

                return (vocalsPath, instrumentalPath);
            }, cancellationToken);
        }

        private void SaveAudio(string filePath, float[,] audio, int sampleRate)
        {
            int channels = audio.GetLength(0);
            int samples = audio.GetLength(1);

            float maxVal = 0f;
            for (int ch = 0; ch < audio.GetLength(0); ch++)
            {
                for (int i = 0; i < audio.GetLength(1); i++)
                {
                    maxVal = Math.Max(maxVal, Math.Abs(audio[ch, i]));
                }
            }

            // Normalize +/-1.0 
            float scale = maxVal > 0.95f ? 0.95f / maxVal : 1.0f;

            var interleaved = new float[samples * channels];
            for (int i = 0; i < samples; i++)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    interleaved[i * channels + ch] = audio[ch, i] * scale;
                }
            }
            Ownaudio.Utilities.WaveFile.WriteFile(filePath, interleaved, sampleRate, channels, 16);
        }

        #endregion
    }

    /// <summary>
    /// Model parameters (internal use)
    /// </summary>
    internal class ModelParameters
    {
        public int DimC { get; } = 4;
        public int DimF { get; set; }
        public int DimT { get; set; }
        public int NFft { get; }
        public int Hop { get; }
        public int NBins { get; }
        public int ChunkSize { get; set; }

        public ModelParameters(int dimF, int dimT, int nFft, int hop = 1024)
        {
            DimF = dimF;
            DimT = (int)Math.Pow(2, dimT);
            NFft = nFft;
            Hop = hop;
            NBins = NFft / 2 + 1;
            ChunkSize = hop * (DimT - 1);
        }
    }

    /// <summary>
    /// Helper extension methods for easier usage
    /// </summary>
    public static class AudioSeparationExtensions
    {
        /// <summary>
        /// Create service with default options
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <returns>Configured AudioSeparationService</returns>
        public static AudioSeparationService CreateDefaultService(string modelPath)
        {
            var options = new SeparationOptions
            {
                ModelPath = modelPath
            };
            return new AudioSeparationService(options);
        }

        /// <summary>
        /// Create service with custom output directory
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <param name="outputDirectory">Output directory path</param>
        /// <returns>Configured AudioSeparationService</returns>
        public static AudioSeparationService CreateService(string modelPath, string outputDirectory)
        {
            var options = new SeparationOptions
            {
                ModelPath = modelPath,
                OutputDirectory = outputDirectory
            };
            return new AudioSeparationService(options);
        }

        /// <summary>
        /// Validate audio file format
        /// </summary>
        /// <param name="filePath">Audio file path</param>
        /// <returns>True if supported format</returns>
        public static bool IsValidAudioFile(string filePath)
        {
            if (!File.Exists(filePath))
                return false;

            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            var supportedFormats = new[] { ".wav", ".mp3", ".flac" };

            return supportedFormats.Contains(extension);
        }

        /// <summary>
        /// Get estimated processing time based on file size
        /// </summary>
        /// <param name="filePath">Audio file path</param>
        /// <returns>Estimated processing time</returns>
        public static TimeSpan EstimateProcessingTime(string filePath)
        {
            if (!File.Exists(filePath))
                return TimeSpan.Zero;

            var fileInfo = new FileInfo(filePath);
            var fileSizeMB = fileInfo.Length / (1024.0 * 1024.0);

            // Rough estimate: ~1-2 minutes per MB for CPU processing
            var estimatedMinutes = fileSizeMB * 1.5;
            return TimeSpan.FromMinutes(Math.Max(0.5, estimatedMinutes));
        }
    }

    /// <summary>
    /// Factory class for creating pre-configured services
    /// </summary>
    public static class AudioSeparationFactory
    {
        /// <summary>
        /// Create service optimized for mobile devices (faster processing)
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <param name="outputDirectory">Output directory path</param>
        /// <returns>Mobile-optimized AudioSeparationService</returns>
        public static AudioSeparationService CreateMobileOptimized(string modelPath, string outputDirectory, bool disableNoiseReduction = false)
        {
            var options = new SeparationOptions
            {
                ModelPath = modelPath,
                OutputDirectory = outputDirectory,
                DisableNoiseReduction = disableNoiseReduction,
                ChunkSizeSeconds = 10, // Smaller chunks for mobile
                Margin = 22050,        // Smaller margin
                NFft = 4096,           // Smaller FFT for speed
                DimT = 7,              // Smaller temporal dimension
                DimF = 1024            // Smaller frequency dimension
            };
            return new AudioSeparationService(options);
        }

        /// <summary>
        /// Create service optimized for desktop (better quality)
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <param name="outputDirectory">Output directory path</param>
        /// <returns>Desktop-optimized AudioSeparationService</returns>
        public static AudioSeparationService CreateDesktopOptimized(string modelPath, string outputDirectory, bool disableNoiseReduction = false)
        {
            var options = new SeparationOptions
            {
                ModelPath = modelPath,
                OutputDirectory = outputDirectory,
                ChunkSizeSeconds = 20, // Larger chunks for better quality
                Margin = 88200,        // Larger margin
                NFft = 8192,           // Larger FFT for quality
                DimT = 9,              // Larger temporal dimension
                DimF = 4096,           // Larger frequency dimension
                DisableNoiseReduction = disableNoiseReduction
            };
            return new AudioSeparationService(options);
        }

        /// <summary>
        /// Create service for batch processing (balanced settings)
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <param name="outputDirectory">Output directory path</param>
        /// <returns>Batch-optimized AudioSeparationService</returns>
        public static AudioSeparationService CreateBatchOptimized(string modelPath, string outputDirectory, bool disableNoiseReduction = true)
        {
            var options = new SeparationOptions
            {
                ModelPath = modelPath,
                OutputDirectory = outputDirectory,
                ChunkSizeSeconds = 15, // Balanced chunk size
                Margin = 44100,        // Standard margin
                NFft = 6144,           // Standard FFT
                DimT = 8,              // Standard temporal dimension
                DimF = 2048,           // Standard frequency dimension
                DisableNoiseReduction = disableNoiseReduction // Faster processing for batch
            };
            return new AudioSeparationService(options);
        }
    }
}