using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Ownaudio;
using Ownaudio.Engines;
using Ownaudio.Sources;
using System.Collections.Concurrent;
using Microsoft.Extensions.ObjectPool;

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
        /// <summary>
        /// Root Mean Square of the original mix audio
        /// </summary>
        public double MixRMS { get; set; }

        /// <summary>
        /// Root Mean Square of the extracted vocals
        /// </summary>
        public double VocalsRMS { get; set; }

        /// <summary>
        /// Root Mean Square of the extracted instrumental
        /// </summary>
        public double InstrumentalRMS { get; set; }

        /// <summary>
        /// Ratio of vocals RMS to mix RMS
        /// </summary>
        public double VocalsMixRatio { get; set; }

        /// <summary>
        /// Ratio of instrumental RMS to mix RMS
        /// </summary>
        public double InstrumentalMixRatio { get; set; }

        /// <summary>
        /// Sample rate of the audio (Hz)
        /// </summary>
        public int SampleRate { get; set; }

        /// <summary>
        /// Number of audio channels
        /// </summary>
        public int Channels { get; set; }

        /// <summary>
        /// Total number of audio samples
        /// </summary>
        public int SampleCount { get; set; }
    }

    /// <summary>
    /// Thread-safe ONNX session wrapper for pooling
    /// </summary>
    public class PooledInferenceSession : IDisposable
    {
        public InferenceSession Session { get; }
        public bool IsDispose { get; set; } = false;

        public PooledInferenceSession(string modelPath, SessionOptions options)
        {
            Session = new InferenceSession(modelPath, options);
        }

        public void Dispose()
        {
            if (!IsDispose)
            {
                Session?.Dispose();
                IsDispose = true;
            }
        }
    }

    /// <summary>
    /// Object pool policy for ONNX sessions
    /// </summary>
    public class InferenceSessionPoolPolicy : IPooledObjectPolicy<PooledInferenceSession>
    {
        private readonly string _modelPath;
        private readonly SessionOptions _sessionOptions;

        public InferenceSessionPoolPolicy(string modelPath, SessionOptions sessionOptions)
        {
            _modelPath = modelPath;
            _sessionOptions = sessionOptions;
        }

        public PooledInferenceSession Create()
        {
            return new PooledInferenceSession(_modelPath, _sessionOptions);
        }

        public bool Return(PooledInferenceSession obj)
        {
            return !obj.IsDispose;
        }
    }

    /// <summary>
    /// Chunk processing result with position information
    /// </summary>
    public class ChunkProcessingResult
    {
        public long Position { get; set; }
        public float[,] ProcessedAudio { get; set; }
        public int OriginalOrder { get; set; }
        public Exception? Error { get; set; }

        public ChunkProcessingResult(long position, float[,] processedAudio, int originalOrder)
        {
            Position = position;
            ProcessedAudio = processedAudio;
            OriginalOrder = originalOrder;
        }
    }

    /// <summary>
    /// Parallel chunk processing configuration
    /// </summary>
    public class ParallelProcessingOptions
    {
        /// <summary>
        /// Maximum degree of parallelism (0 = auto-detect)
        /// </summary>
        public int MaxDegreeOfParallelism { get; set; } = 0;

        /// <summary>
        /// ONNX session pool size
        /// </summary>
        public int SessionPoolSize { get; set; } = 0;

        /// <summary>
        /// Enable memory pressure monitoring
        /// </summary>
        public bool EnableMemoryPressureMonitoring { get; set; } = true;

        /// <summary>
        /// Memory pressure threshold (bytes)
        /// </summary>
        public long MemoryPressureThreshold { get; set; } = 2_000_000_000; // 2GB

        /// <summary>
        /// Chunk queue capacity
        /// </summary>
        public int ChunkQueueCapacity { get; set; } = 10;
    }

    /// <summary>
    /// Main audio separation service with parallel processing support
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

        /// <summary>
        /// Configuration options for separation
        /// </summary>
        private readonly SeparationOptions _options;

        /// <summary>
        /// Model parameters for STFT processing
        /// </summary>
        private ModelParameters _modelParams;

        /// <summary>
        /// ONNX Runtime session for model inference (traditional mode)
        /// </summary>
        private InferenceSession? _onnxSession;

        /// <summary>
        /// ONNX session pool for parallel processing
        /// </summary>
        private ObjectPool<PooledInferenceSession>? _sessionPool;

        /// <summary>
        /// Parallel processing configuration
        /// </summary>
        private ParallelProcessingOptions? _parallelOptions;

        /// <summary>
        /// Semaphore for controlling concurrent session usage
        /// </summary>
        private SemaphoreSlim? _sessionSemaphore;

        /// <summary>
        /// Memory monitoring timer
        /// </summary>
        private Timer? _memoryMonitorTimer;

        /// <summary>
        /// Current memory pressure flag
        /// </summary>
        private volatile bool _isMemoryPressureHigh = false;

        /// <summary>
        /// Flag indicating if the object has been disposed
        /// </summary>
        private bool _disposed = false;

        /// <summary>
        /// Target sample rate for audio processing
        /// </summary>
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
        /// Initialize the ONNX model session and auto-detect model dimensions (traditional mode)
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
        /// Initialize parallel processing with session pool
        /// </summary>
        /// <param name="parallelOptions">Parallel processing configuration</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task InitializeParallelAsync(
            ParallelProcessingOptions? parallelOptions = null,
            CancellationToken cancellationToken = default)
        {
            _parallelOptions = parallelOptions ?? GetDefaultParallelOptions();

            await Task.Run(() =>
            {
                if (!File.Exists(_options.ModelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {_options.ModelPath}");
                }

                // Create session options
                var sessionOptions = new SessionOptions
                {
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };

                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA();
                    Console.WriteLine("CUDA execution provider enabled for parallel processing.");
                }
                catch
                {
                    sessionOptions.AppendExecutionProvider_CPU();
                    Console.WriteLine("Using CPU execution provider for parallel processing.");
                }

                // Create session pool
                var poolPolicy = new InferenceSessionPoolPolicy(_options.ModelPath, sessionOptions);
                _sessionPool = new DefaultObjectPool<PooledInferenceSession>(
                    poolPolicy, _parallelOptions.SessionPoolSize);

                // Create semaphore for session control
                _sessionSemaphore = new SemaphoreSlim(
                    _parallelOptions.SessionPoolSize,
                    _parallelOptions.SessionPoolSize);

                // Initialize one session for auto-detection
                using var testSession = _sessionPool.Get();
                _onnxSession = testSession.Session;
                AutoDetectModelDimensions();
                _onnxSession = null; // Clear reference, use pool from now on

                // Start memory monitoring if enabled
                if (_parallelOptions.EnableMemoryPressureMonitoring)
                {
                    StartMemoryMonitoring();
                }

                Console.WriteLine($"Parallel processing initialized: " +
                    $"Sessions={_parallelOptions.SessionPoolSize}, " +
                    $"MaxParallelism={_parallelOptions.MaxDegreeOfParallelism}");

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
            // Check if either traditional or parallel processing is initialized
            if (_onnxSession == null && _sessionPool == null)
                throw new InvalidOperationException("Service not initialized. Call InitializeAsync or InitializeParallelAsync first.");

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

                // Use parallel processing if available, otherwise fall back to traditional
                var separated = _sessionPool != null
                    ? await ProcessAudioParallelAsync(mix, cancellationToken)
                    : await ProcessAudioAsync(mix, cancellationToken);

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
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        #endregion

        #region Private Methods - Initialization

        /// <summary>
        /// Automatically detect model dimensions from ONNX metadata
        /// </summary>
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

        /// <summary>
        /// Get default parallel processing options based on system capabilities
        /// </summary>
        /// <returns>Default parallel processing options</returns>
        private ParallelProcessingOptions GetDefaultParallelOptions()
        {
            int cpuCount = Environment.ProcessorCount;
            long totalMemory = GC.GetTotalMemory(false);

            return new ParallelProcessingOptions
            {
                MaxDegreeOfParallelism = Math.Max(1, cpuCount / 2), // Conservative approach
                SessionPoolSize = Math.Min(4, Math.Max(2, cpuCount / 2)), // 2-4 sessions
                EnableMemoryPressureMonitoring = totalMemory > 1_000_000_000, // Only if >1GB available
                MemoryPressureThreshold = totalMemory / 2, // 50% of available memory
                ChunkQueueCapacity = cpuCount * 2
            };
        }

        /// <summary>
        /// Start memory monitoring for adaptive parallelism
        /// </summary>
        private void StartMemoryMonitoring()
        {
            _memoryMonitorTimer = new Timer(MonitorMemoryPressure, null,
                TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(5));
        }

        /// <summary>
        /// Monitor memory pressure and adjust processing accordingly
        /// </summary>
        /// <param name="state">Timer state (unused)</param>
        private void MonitorMemoryPressure(object? state)
        {
            try
            {
                long currentMemory = GC.GetTotalMemory(false);
                _isMemoryPressureHigh = currentMemory > _parallelOptions!.MemoryPressureThreshold;

                if (_isMemoryPressureHigh)
                {
                    // Force garbage collection under memory pressure
                    GC.Collect(1, GCCollectionMode.Optimized);
                }
            }
            catch
            {
                // Ignore monitoring errors
            }
        }

        #endregion

        #region Private Methods - Audio Processing

        /// <summary>
        /// Report progress to subscribers
        /// </summary>
        /// <param name="progress">Progress information</param>
        private void ReportProgress(SeparationProgress progress)
        {
            ProgressChanged?.Invoke(this, progress);
        }

        /// <summary>
        /// Load and prepare audio file for processing
        /// </summary>
        /// <param name="filePath">Path to audio file</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Audio data as 2D array [channels, samples]</returns>
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

                _manager.AddOutputSource(filePath);

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

        /// <summary>
        /// Process audio using model inference with chunking (traditional mode)
        /// </summary>
        /// <param name="mix">Input audio mix</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Separated audio</returns>
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

        /// <summary>
        /// Process audio chunks in parallel
        /// </summary>
        /// <param name="mix">Input audio mix</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Processed audio</returns>
        private async Task<float[,]> ProcessAudioParallelAsync(
            float[,] mix, CancellationToken cancellationToken)
        {
            if (_sessionPool == null)
            {
                throw new InvalidOperationException(
                    "Parallel processing not initialized. Call InitializeParallelAsync first.");
            }

            return await Task.Run(() =>
            {
                int samples = mix.GetLength(1);
                int margin = _options.Margin;
                int chunkSize = _options.ChunkSizeSeconds * TargetSampleRate;

                if (margin == 0) throw new ArgumentException("Margin cannot be zero!");
                if (chunkSize != 0 && margin > chunkSize) margin = chunkSize;
                if (_options.ChunkSizeSeconds == 0 || samples < chunkSize) chunkSize = samples;

                var chunks = CreateChunksWithOrder(mix, chunkSize, margin);
                return ProcessChunksParallel(chunks, margin, cancellationToken);
            }, cancellationToken);
        }

        #endregion

        #region Private Methods - Chunk Processing

        /// <summary>
        /// Create audio chunks with overlapping margins
        /// </summary>
        /// <param name="mix">Input audio mix</param>
        /// <param name="chunkSize">Size of each chunk</param>
        /// <param name="margin">Overlap margin size</param>
        /// <returns>Dictionary of chunks indexed by position</returns>
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

        /// <summary>
        /// Create chunks with ordering information for parallel processing
        /// </summary>
        /// <param name="mix">Input audio mix</param>
        /// <param name="chunkSize">Size of each chunk</param>
        /// <param name="margin">Overlap margin size</param>
        /// <returns>Dictionary of chunks with position and order info</returns>
        private Dictionary<long, (float[,] audio, int order)> CreateChunksWithOrder(
            float[,] mix, int chunkSize, int margin)
        {
            var chunks = new Dictionary<long, (float[,], int)>();
            int samples = mix.GetLength(1);
            long counter = -1;
            int order = 0;

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

                chunks[skip] = (segment, order++);
                if (end == samples) break;
            }

            return chunks;
        }

        /// <summary>
        /// Process all audio chunks through the model (traditional mode)
        /// </summary>
        /// <param name="chunks">Audio chunks to process</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Concatenated processed audio</returns>
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

        /// <summary>
        /// Process chunks in parallel with proper ordering and error handling
        /// </summary>
        /// <param name="chunks">Chunks to process with order information</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Concatenated processed audio</returns>
        private float[,] ProcessChunksParallel(
            Dictionary<long, (float[,] audio, int order)> chunks,
            int margin,
            CancellationToken cancellationToken)
        {
            var keys = chunks.Keys.ToList();
            int totalChunks = chunks.Count;
            var results = new ConcurrentDictionary<int, ChunkProcessingResult>();
            var exceptions = new ConcurrentQueue<Exception>();

            ReportProgress(new SeparationProgress
            {
                Status = "Processing chunks in parallel...",
                TotalChunks = totalChunks,
                ProcessedChunks = 0,
                OverallProgress = 20
            });

            // Determine optimal parallelism
            int maxParallelism = GetOptimalParallelism();
            var parallelOptions = new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = maxParallelism
            };

            // Process chunks in parallel
            var processedCount = 0;
            var lockObject = new object();

            try
            {
                Parallel.ForEach(chunks, parallelOptions, kvp =>
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Check memory pressure and throttle if needed
                    if (_isMemoryPressureHigh)
                    {
                        Thread.Sleep(100); // Brief pause during memory pressure
                    }

                    try
                    {
                        var result = ProcessSingleChunkParallel(
                            kvp.Value.audio, kvp.Key, keys, margin, kvp.Value.order);

                        results[kvp.Value.order] = result;

                        // Update progress thread-safely
                        lock (lockObject)
                        {
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
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                        var errorResult = new ChunkProcessingResult(kvp.Key, new float[2, 0], kvp.Value.order)
                        {
                            Error = ex
                        };
                        results[kvp.Value.order] = errorResult;
                    }
                });
            }
            catch (AggregateException aggEx)
            {
                foreach (var ex in aggEx.InnerExceptions)
                {
                    exceptions.Enqueue(ex);
                }
            }

            // Check for errors
            if (!exceptions.IsEmpty)
            {
                var firstError = exceptions.TryDequeue(out var error) ? error :
                    new Exception("Unknown error during parallel processing");
                throw new AggregateException("Errors occurred during parallel chunk processing",
                    exceptions.ToArray().Concat(new[] { firstError }));
            }

            // Sort results by original order and concatenate
            var orderedResults = results.Values
                .OrderBy(r => r.OriginalOrder)
                .Where(r => r.Error == null)
                .Select(r => r.ProcessedAudio)
                .ToList();

            return ConcatenateChunks(orderedResults);
        }

        /// <summary>
        /// Get optimal parallelism degree considering current system state
        /// </summary>
        /// <returns>Optimal degree of parallelism</returns>
        private int GetOptimalParallelism()
        {
            int baseDegree = _parallelOptions?.MaxDegreeOfParallelism ?? 0;

            if (baseDegree == 0)
            {
                baseDegree = Math.Max(1, Environment.ProcessorCount / 2);
            }

            // Reduce parallelism under memory pressure
            if (_isMemoryPressureHigh)
            {
                return Math.Max(1, baseDegree / 2);
            }

            return baseDegree;
        }

        #endregion

        #region Private Methods - Single Chunk Processing

        /// <summary>
        /// Process a single audio chunk through STFT, model inference, and ISTFT (traditional mode)
        /// </summary>
        /// <param name="mixChunk">Audio chunk to process</param>
        /// <param name="chunkKey">Position key of the chunk</param>
        /// <param name="allKeys">All chunk position keys</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <returns>Processed audio chunk</returns>
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

        /// <summary>
        /// Process single chunk with session pooling
        /// </summary>
        /// <param name="mixChunk">Audio chunk to process</param>
        /// <param name="chunkKey">Position key of the chunk</param>
        /// <param name="allKeys">All chunk position keys</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <param name="order">Original order for sorting</param>
        /// <returns>Processing result with order information</returns>
        private ChunkProcessingResult ProcessSingleChunkParallel(
            float[,] mixChunk, long chunkKey, List<long> allKeys, int margin, int order)
        {
            if (_sessionPool == null || _sessionSemaphore == null)
            {
                throw new InvalidOperationException("Session pool not initialized");
            }

            // Wait for available session
            _sessionSemaphore.Wait();
            PooledInferenceSession? pooledSession = null;

            try
            {
                pooledSession = _sessionPool.Get();

                // Use the pooled session for processing
                var processedAudio = ProcessSingleChunkWithSession(
                    mixChunk, chunkKey, allKeys, margin, pooledSession.Session);

                return new ChunkProcessingResult(chunkKey, processedAudio, order);
            }
            finally
            {
                // Return session to pool
                if (pooledSession != null)
                {
                    _sessionPool.Return(pooledSession);
                }
                _sessionSemaphore.Release();
            }
        }

        /// <summary>
        /// Process single chunk with specific ONNX session
        /// </summary>
        /// <param name="mixChunk">Audio chunk to process</param>
        /// <param name="chunkKey">Position key of the chunk</param>
        /// <param name="allKeys">All chunk position keys</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <param name="session">ONNX session to use</param>
        /// <returns>Processed audio chunk</returns>
        private float[,] ProcessSingleChunkWithSession(
            float[,] mixChunk, long chunkKey, List<long> allKeys, int margin, InferenceSession session)
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

            // STFT -> Model -> ISTFT with specific session
            var stftTensor = ComputeStft(mixWaves);
            var outputTensor = RunModelInferenceWithSession(stftTensor, session);
            var resultWaves = ComputeIstft(outputTensor);

            // Extract and apply margin
            var result = ExtractSignal(resultWaves, nSample, trim, genSize);
            return ApplyMargin(result, chunkKey, allKeys, margin);
        }

        #endregion

        #region Private Methods - STFT/ISTFT Processing

        /// <summary>
        /// Compute Short-Time Fourier Transform (STFT) for audio waves
        /// </summary>
        /// <param name="mixWaves">Audio waves to transform</param>
        /// <returns>STFT tensor in model input format</returns>
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

        /// <summary>
        /// Compute Inverse Short-Time Fourier Transform (ISTFT) from spectrum
        /// </summary>
        /// <param name="spectrum">Frequency domain spectrum</param>
        /// <returns>Time domain audio waves</returns>
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

        #endregion

        #region Private Methods - Model Inference

        /// <summary>
        /// Run model inference on STFT tensor (traditional mode)
        /// </summary>
        /// <param name="stftTensor">Input STFT tensor</param>
        /// <returns>Output tensor from model</returns>
        private Tensor<float> RunModelInference(DenseTensor<float> stftTensor)
        {
            if (_onnxSession == null)
                throw new InvalidOperationException("ONNX session not initialized");

            return RunModelInferenceWithSession(stftTensor, _onnxSession);
        }

        /// <summary>
        /// Run model inference with specific session
        /// </summary>
        /// <param name="stftTensor">Input STFT tensor</param>
        /// <param name="session">ONNX session to use</param>
        /// <returns>Output tensor from model</returns>
        private Tensor<float> RunModelInferenceWithSession(DenseTensor<float> stftTensor, InferenceSession session)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", stftTensor) };

            if (!_options.DisableNoiseReduction)
            {
                // Denoise logic with specific session
                var stftTensorNeg = new DenseTensor<float>(stftTensor.Dimensions);
                for (int idx = 0; idx < stftTensor.Length; idx++)
                {
                    stftTensorNeg.SetValue(idx, -stftTensor.GetValue(idx));
                }
                var inputsNeg = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", stftTensorNeg) };

                using var outputs = session.Run(inputs);
                using var outputsNeg = session.Run(inputsNeg);

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
                using var outputs = session.Run(inputs);
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

        #endregion

        #region Private Methods - Signal Processing

        /// <summary>
        /// Extract processed signal from wave frames
        /// </summary>
        /// <param name="waves">Processed wave frames</param>
        /// <param name="nSample">Number of samples to extract</param>
        /// <param name="trim">Trim size from frame edges</param>
        /// <param name="genSize">Generation size per frame</param>
        /// <returns>Extracted signal</returns>
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

        /// <summary>
        /// Apply margin trimming to processed signal
        /// </summary>
        /// <param name="signal">Processed signal</param>
        /// <param name="chunkKey">Current chunk position key</param>
        /// <param name="allKeys">All chunk position keys</param>
        /// <param name="margin">Margin size for trimming</param>
        /// <returns>Margin-trimmed signal</returns>
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

        /// <summary>
        /// Concatenate processed audio chunks into single array
        /// </summary>
        /// <param name="chunks">List of processed audio chunks</param>
        /// <returns>Concatenated audio array</returns>
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

        #endregion

        #region Private Methods - Statistics and File I/O

        /// <summary>
        /// Calculate RMS and ratio statistics for audio signals
        /// </summary>
        /// <param name="mix">Original mix audio</param>
        /// <param name="vocals">Extracted vocals</param>
        /// <param name="instrumental">Extracted instrumental</param>
        /// <returns>Calculated audio statistics</returns>
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

        /// <summary>
        /// Calculate Root Mean Square (RMS) value of audio signal
        /// </summary>
        /// <param name="audio">Audio signal to analyze</param>
        /// <returns>RMS value</returns>
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

        /// <summary>
        /// Save separated audio results to files
        /// </summary>
        /// <param name="filename">Base filename for output files</param>
        /// <param name="vocals">Vocals audio data</param>
        /// <param name="instrumental">Instrumental audio data</param>
        /// <param name="sampleRate">Audio sample rate</param>
        /// <param name="modelName">Name of the model used</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Tuple of vocals and instrumental file paths</returns>
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

        /// <summary>
        /// Save audio data to WAV file with normalization
        /// </summary>
        /// <param name="filePath">Output file path</param>
        /// <param name="audio">Audio data to save</param>
        /// <param name="sampleRate">Audio sample rate</param>
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

        #region Dispose Pattern

        /// <summary>
        /// Dispose resources
        /// </summary>
        /// <param name="disposing">True if disposing</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _memoryMonitorTimer?.Dispose();
                _sessionSemaphore?.Dispose();

                // Dispose all sessions in pool
                if (_sessionPool is IDisposable disposablePool)
                {
                    disposablePool.Dispose();
                }

                // Dispose original session if it exists
                _onnxSession?.Dispose();
                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// Model parameters for STFT processing (internal use)
    /// </summary>
    internal class ModelParameters
    {
        /// <summary>
        /// Number of channels (always 4 for complex stereo)
        /// </summary>
        public int DimC { get; } = 4;

        /// <summary>
        /// Frequency dimension size
        /// </summary>
        public int DimF { get; set; }

        /// <summary>
        /// Time dimension size (calculated as power of 2)
        /// </summary>
        public int DimT { get; set; }

        /// <summary>
        /// FFT size
        /// </summary>
        public int NFft { get; }

        /// <summary>
        /// Hop size for STFT
        /// </summary>
        public int Hop { get; }

        /// <summary>
        /// Number of frequency bins (NFft/2 + 1)
        /// </summary>
        public int NBins { get; }

        /// <summary>
        /// Processing chunk size in samples
        /// </summary>
        public int ChunkSize { get; set; }

        /// <summary>
        /// Initialize model parameters
        /// </summary>
        /// <param name="dimF">Frequency dimension</param>
        /// <param name="dimT">Time dimension (as power of 2)</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="hop">Hop size (default 1024)</param>
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
        /// <param name="disableNoiseReduction">Whether to disable noise reduction for speed</param>
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
        /// <param name="disableNoiseReduction">Whether to disable noise reduction</param>
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
        /// <param name="disableNoiseReduction">Whether to disable noise reduction for faster batch processing</param>
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

        /// <summary>
        /// Create service with parallel processing optimized for system capabilities
        /// </summary>
        /// <param name="modelPath">Path to ONNX model file</param>
        /// <param name="outputDirectory">Output directory path</param>
        /// <param name="systemCores">Number of CPU cores</param>
        /// <param name="availableMemoryGB">Available memory in GB</param>
        /// <returns>System-optimized AudioSeparationService with parallel options</returns>
        public static (AudioSeparationService service, ParallelProcessingOptions parallelOptions) CreateSystemOptimized(
            string modelPath, string outputDirectory, int systemCores, double availableMemoryGB)
        {
            SeparationOptions separationOptions;
            ParallelProcessingOptions parallelOptions;

            if (availableMemoryGB > 16 && systemCores >= 12)
            {
                // High-end workstation
                separationOptions = new SeparationOptions
                {
                    ModelPath = modelPath,
                    OutputDirectory = outputDirectory,
                    ChunkSizeSeconds = 30,
                    Margin = 88200,
                    NFft = 8192,
                    DimT = 9,
                    DimF = 4096,
                    DisableNoiseReduction = false
                };

                parallelOptions = new ParallelProcessingOptions
                {
                    MaxDegreeOfParallelism = Math.Min(8, systemCores),
                    SessionPoolSize = Math.Min(6, systemCores / 2),
                    EnableMemoryPressureMonitoring = true,
                    MemoryPressureThreshold = (long)(availableMemoryGB * 0.7 * 1024 * 1024 * 1024),
                    ChunkQueueCapacity = systemCores * 2
                };
            }
            else if (availableMemoryGB > 8 && systemCores >= 8)
            {
                // High-end desktop
                separationOptions = new SeparationOptions
                {
                    ModelPath = modelPath,
                    OutputDirectory = outputDirectory,
                    ChunkSizeSeconds = 20,
                    Margin = 66150,
                    NFft = 6144,
                    DimT = 8,
                    DimF = 3072,
                    DisableNoiseReduction = false
                };

                parallelOptions = new ParallelProcessingOptions
                {
                    MaxDegreeOfParallelism = Math.Min(6, systemCores * 3 / 4),
                    SessionPoolSize = Math.Min(4, systemCores / 2),
                    EnableMemoryPressureMonitoring = true,
                    MemoryPressureThreshold = (long)(availableMemoryGB * 0.6 * 1024 * 1024 * 1024),
                    ChunkQueueCapacity = systemCores
                };
            }
            else if (availableMemoryGB > 4 && systemCores >= 4)
            {
                // Mid-range system
                separationOptions = new SeparationOptions
                {
                    ModelPath = modelPath,
                    OutputDirectory = outputDirectory,
                    ChunkSizeSeconds = 15,
                    Margin = 44100,
                    NFft = 6144,
                    DimT = 8,
                    DimF = 2048,
                    DisableNoiseReduction = true
                };

                parallelOptions = new ParallelProcessingOptions
                {
                    MaxDegreeOfParallelism = Math.Min(4, systemCores / 2),
                    SessionPoolSize = 3,
                    EnableMemoryPressureMonitoring = true,
                    MemoryPressureThreshold = (long)(availableMemoryGB * 0.5 * 1024 * 1024 * 1024),
                    ChunkQueueCapacity = systemCores
                };
            }
            else
            {
                // Low-end or mobile system
                separationOptions = new SeparationOptions
                {
                    ModelPath = modelPath,
                    OutputDirectory = outputDirectory,
                    ChunkSizeSeconds = 10,
                    Margin = 22050,
                    NFft = 4096,
                    DimT = 7,
                    DimF = 1024,
                    DisableNoiseReduction = true
                };

                parallelOptions = new ParallelProcessingOptions
                {
                    MaxDegreeOfParallelism = Math.Max(1, systemCores / 2),
                    SessionPoolSize = 2,
                    EnableMemoryPressureMonitoring = true,
                    MemoryPressureThreshold = (long)(availableMemoryGB * 0.4 * 1024 * 1024 * 1024),
                    ChunkQueueCapacity = Math.Max(2, systemCores)
                };
            }

            var service = new AudioSeparationService(separationOptions);
            return (service, parallelOptions);
        }
    }
}