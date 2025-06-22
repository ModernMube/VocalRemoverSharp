using OwnSeparator.Core;
using System.Diagnostics;

namespace OwnSeparator.BasicConsole
{
    /// <summary>
    /// Main program class for the OwnSeparator console application with parallel processing
    /// </summary>
    class Program
    {
        /// <summary>
        /// Main entry point of the application
        /// </summary>
        /// <param name="args">Command line arguments</param>
        /// <returns>Task representing the asynchronous operation</returns>
        static async Task Main(string[] args)
        {
            Console.WriteLine("OwnSeparator Audio Separation - Parallel Version");
            Console.WriteLine("===============================================");

            try
            {
                var systemInfo = AnalyzeSystemCapabilities();
                DisplaySystemInfo(systemInfo);

                var separationOptions = CreateOptimalSeparationOptions(systemInfo);

                var parallelOptions = CreateOptimalParallelOptions(systemInfo);

                Console.WriteLine($"\nUsing configuration: {GetConfigurationName(separationOptions)}");
                Console.WriteLine($"Parallel processing: {parallelOptions.MaxDegreeOfParallelism} threads, " +
                                $"{parallelOptions.SessionPoolSize} ONNX sessions");

                //Create service optimized for mobile devices (faster processing)
                separationOptions = new SeparationOptions
                {
                    ModelPath = "models/OWN_INST_BEST.ONNX",
                    OutputDirectory = "output",
                    ChunkSizeSeconds = 10,
                    Margin = 22050,
                    NFft = 4096,
                    DimT = 7,
                    DimF = 1024,
                    DisableNoiseReduction = true
                };

                var service = new AudioSeparationService(separationOptions);

                service.ProgressChanged += OnProgressChanged;
                service.ProcessingStarted += OnProcessingStarted;
                service.ProcessingCompleted += OnProcessingCompleted;
                service.ErrorOccurred += OnErrorOccurred;

                Console.WriteLine("\nInitializing parallel processing...");
                var initStopwatch = Stopwatch.StartNew();

                await service.InitializeParallelAsync(parallelOptions);

                initStopwatch.Stop();
                Console.WriteLine($"Initialization completed in {initStopwatch.ElapsedMilliseconds}ms");

                await ProcessSingleFile(service, @"input/audio.mp3");

                if (args.Length > 0 && args[0] == "--batch")
                {
                    await ProcessBatchFiles(service);
                }

                DisplayMemoryStatistics();

                service.Dispose();
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine($"❌ File not found: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ An error occurred: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"   Inner exception: {ex.InnerException.Message}");
                }
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Analyze system capabilities for optimal configuration
        /// </summary>
        /// <returns>System information structure</returns>
        private static SystemInfo AnalyzeSystemCapabilities()
        {
            var totalMemory = GC.GetTotalMemory(false);
            var cpuCount = Environment.ProcessorCount;

            var estimatedTotalRAM = totalMemory * 10; // Rough estimation

            return new SystemInfo
            {
                CpuCores = cpuCount,
                AvailableMemory = totalMemory,
                EstimatedTotalRAM = estimatedTotalRAM,
                Is64Bit = Environment.Is64BitProcess,
                HasHighMemory = estimatedTotalRAM > 8_000_000_000, // >8GB
                HasManyCores = cpuCount >= 8
            };
        }

        /// <summary>
        /// Create optimal separation options based on system info
        /// </summary>
        /// <param name="systemInfo">System capability information</param>
        /// <returns>Optimized separation options</returns>
        private static SeparationOptions CreateOptimalSeparationOptions(SystemInfo systemInfo)
        {
            if (systemInfo.HasHighMemory && systemInfo.HasManyCores)
            {
                return new SeparationOptions
                {
                    ModelPath = "models/OWN_INST_BEST.ONNX",
                    OutputDirectory = "output",
                    ChunkSizeSeconds = 25,
                    Margin = 88200,
                    NFft = 8192,
                    DimT = 9,
                    DimF = 4096,
                    DisableNoiseReduction = false
                };
            }
            else if (systemInfo.HasManyCores || systemInfo.HasHighMemory)
            {
                return new SeparationOptions
                {
                    ModelPath = "models/OWN_INST_BEST.ONNX",
                    OutputDirectory = "output",
                    ChunkSizeSeconds = 15,
                    Margin = 44100,
                    NFft = 6144,
                    DimT = 8,
                    DimF = 2048,
                    DisableNoiseReduction = true
                };
            }
            else
            {
                return new SeparationOptions
                {
                    ModelPath = "models/OWN_INST_BEST.ONNX",
                    OutputDirectory = "output",
                    ChunkSizeSeconds = 10,
                    Margin = 22050,
                    NFft = 4096,
                    DimT = 7,
                    DimF = 1024,
                    DisableNoiseReduction = true
                };
            }
        }

        /// <summary>
        /// Create optimal parallel processing options based on system info
        /// </summary>
        /// <param name="systemInfo">System capability information</param>
        /// <returns>Optimized parallel processing options</returns>
        private static ParallelProcessingOptions CreateOptimalParallelOptions(SystemInfo systemInfo)
        {
            int maxParallelism;
            int sessionPoolSize;
            long memoryThreshold;

            if (systemInfo.HasHighMemory && systemInfo.HasManyCores)
            {
                maxParallelism = Math.Min(8, systemInfo.CpuCores);
                sessionPoolSize = Math.Min(6, systemInfo.CpuCores / 2);
                memoryThreshold = systemInfo.EstimatedTotalRAM / 3; // Use up to 1/3 of RAM
            }
            else if (systemInfo.HasManyCores)
            {
                maxParallelism = Math.Min(4, systemInfo.CpuCores / 2);
                sessionPoolSize = Math.Min(3, systemInfo.CpuCores / 3);
                memoryThreshold = systemInfo.EstimatedTotalRAM / 2; // Use up to 1/2 of RAM
            }
            else if (systemInfo.HasHighMemory)
            {
                maxParallelism = Math.Min(4, systemInfo.CpuCores);
                sessionPoolSize = 3;
                memoryThreshold = systemInfo.EstimatedTotalRAM / 4; // Use up to 1/4 of RAM
            }
            else
            {
                maxParallelism = Math.Max(1, systemInfo.CpuCores / 2);
                sessionPoolSize = 2;
                memoryThreshold = systemInfo.AvailableMemory * 2; // Conservative
            }

            return new ParallelProcessingOptions
            {
                MaxDegreeOfParallelism = maxParallelism,
                SessionPoolSize = sessionPoolSize,
                EnableMemoryPressureMonitoring = true,
                MemoryPressureThreshold = memoryThreshold,
                ChunkQueueCapacity = maxParallelism * 2
            };
        }

        /// <summary>
        /// Process a single audio file with detailed progress tracking
        /// </summary>
        /// <param name="service">Audio separation service</param>
        /// <param name="inputFile">Input file path</param>
        private static async Task ProcessSingleFile(AudioSeparationService service, string inputFile)
        {
            if (!File.Exists(inputFile))
            {
                Console.WriteLine($"⚠️  Input file not found: {inputFile}");
                Console.WriteLine("   Creating example input directory...");
                Directory.CreateDirectory("input");
                Console.WriteLine("   Please place your audio file at: input/audio.mp3");
                return;
            }

            Console.WriteLine($"\n🎵 Processing: {Path.GetFileName(inputFile)}");

            var totalStopwatch = Stopwatch.StartNew();
            var result = await service.SeparateAsync(inputFile);
            totalStopwatch.Stop();

            Console.WriteLine($"\n✅ Processing completed!");
            Console.WriteLine($"📁 Vocals file: {result.VocalsPath}");
            Console.WriteLine($"📁 Instrumental file: {result.InstrumentalPath}");
            Console.WriteLine($"⏱️  Total processing time: {totalStopwatch.Elapsed:mm\\:ss\\.ff}");

            DisplayAudioStatistics(result.Statistics);
        }

        /// <summary>
        /// Process multiple files for batch demonstration
        /// </summary>
        /// <param name="service">Audio separation service</param>
        private static async Task ProcessBatchFiles(AudioSeparationService service)
        {
            var inputDir = "input";
            if (!Directory.Exists(inputDir))
            {
                Console.WriteLine("⚠️  Input directory not found for batch processing");
                return;
            }

            var audioFiles = Directory.GetFiles(inputDir, "*.*")
                .Where(f => AudioSeparationExtensions.IsValidAudioFile(f))
                .ToList();

            if (!audioFiles.Any())
            {
                Console.WriteLine("⚠️  No valid audio files found for batch processing");
                return;
            }

            Console.WriteLine($"\n📦 Starting batch processing of {audioFiles.Count} files...");

            var batchStopwatch = Stopwatch.StartNew();
            var results = await service.SeparateMultipleAsync(audioFiles);
            batchStopwatch.Stop();

            Console.WriteLine($"\n✅ Batch processing completed!");
            Console.WriteLine($"📊 Processed {results.Count} files in {batchStopwatch.Elapsed:mm\\:ss\\.ff}");
            Console.WriteLine($"📊 Average time per file: {TimeSpan.FromMilliseconds(batchStopwatch.ElapsedMilliseconds / results.Count):mm\\:ss\\.ff}");
        }

        /// <summary>
        /// Display system information
        /// </summary>
        /// <param name="systemInfo">System information to display</param>
        private static void DisplaySystemInfo(SystemInfo systemInfo)
        {
            Console.WriteLine($"\n💻 System Information:");
            Console.WriteLine($"   CPU Cores: {systemInfo.CpuCores}");
            Console.WriteLine($"   Available Memory: {systemInfo.AvailableMemory / 1024 / 1024:N0} MB");
            Console.WriteLine($"   Estimated Total RAM: {systemInfo.EstimatedTotalRAM / 1024 / 1024 / 1024:N1} GB");
            Console.WriteLine($"   64-bit Process: {systemInfo.Is64Bit}");
            Console.WriteLine($"   Performance Class: {GetPerformanceClass(systemInfo)}");
        }

        /// <summary>
        /// Display audio processing statistics
        /// </summary>
        /// <param name="stats">Audio statistics to display</param>
        private static void DisplayAudioStatistics(AudioStatistics stats)
        {
            Console.WriteLine($"\n📊 Audio Statistics:");
            Console.WriteLine($"   Sample Rate: {stats.SampleRate:N0} Hz");
            Console.WriteLine($"   Channels: {stats.Channels}");
            Console.WriteLine($"   Duration: {TimeSpan.FromSeconds(stats.SampleCount / (double)stats.SampleRate):mm\\:ss}");
            Console.WriteLine($"   Mix RMS: {stats.MixRMS:F6}");
            Console.WriteLine($"   Vocals RMS: {stats.VocalsRMS:F6} ({stats.VocalsMixRatio:P1} of mix)");
            Console.WriteLine($"   Instrumental RMS: {stats.InstrumentalRMS:F6} ({stats.InstrumentalMixRatio:P1} of mix)");
        }

        /// <summary>
        /// Display current memory statistics
        /// </summary>
        private static void DisplayMemoryStatistics()
        {
            var currentMemory = GC.GetTotalMemory(false);
            var gen0 = GC.CollectionCount(0);
            var gen1 = GC.CollectionCount(1);
            var gen2 = GC.CollectionCount(2);

            Console.WriteLine($"\n🧠 Memory Statistics:");
            Console.WriteLine($"   Current Memory: {currentMemory / 1024 / 1024:N1} MB");
            Console.WriteLine($"   GC Collections - Gen0: {gen0}, Gen1: {gen1}, Gen2: {gen2}");
        }

        /// <summary>
        /// Get configuration name based on settings
        /// </summary>
        /// <param name="options">Separation options</param>
        /// <returns>Configuration name</returns>
        private static string GetConfigurationName(SeparationOptions options)
        {
            if (options.NFft >= 8192 && options.DimF >= 4096)
                return "Desktop Quality (High Resource)";
            else if (options.NFft >= 6144 && options.DimF >= 2048)
                return "Balanced Performance";
            else
                return "Mobile Optimized (Low Resource)";
        }

        /// <summary>
        /// Get performance class description
        /// </summary>
        /// <param name="systemInfo">System information</param>
        /// <returns>Performance class string</returns>
        private static string GetPerformanceClass(SystemInfo systemInfo)
        {
            if (systemInfo.HasHighMemory && systemInfo.HasManyCores)
                return "High-End Desktop";
            else if (systemInfo.HasManyCores)
                return "Multi-Core System";
            else if (systemInfo.HasHighMemory)
                return "High-Memory System";
            else
                return "Standard/Mobile";
        }

        #region Event Handlers

        /// <summary>
        /// Handle progress change events with enhanced display
        /// </summary>
        private static void OnProgressChanged(object? sender, SeparationProgress progress)
        {
            Console.Write($"\r🔄 {progress.Status}: {progress.OverallProgress:F1}%");

            if (progress.TotalChunks > 0)
            {
                Console.Write($" ({progress.ProcessedChunks}/{progress.TotalChunks} chunks)");
            }

            if (progress.ChunkProgress > 0)
            {
                Console.Write($" [Chunk: {progress.ChunkProgress:F0}%]");
            }

            var barWidth = 20;
            var filled = (int)(progress.OverallProgress / 100.0 * barWidth);
            var bar = "[" + new string('█', filled) + new string('░', barWidth - filled) + "]";
            Console.Write($" {bar}");
        }

        /// <summary>
        /// Handle processing started events
        /// </summary>
        private static void OnProcessingStarted(object? sender, string inputFile)
        {
            Console.WriteLine($"\n🚀 Started processing: {Path.GetFileName(inputFile)}");

            // Estimate processing time
            var estimatedTime = AudioSeparationExtensions.EstimateProcessingTime(inputFile);
            if (estimatedTime > TimeSpan.Zero)
            {
                Console.WriteLine($"⏱️  Estimated time: ~{estimatedTime:mm\\:ss}");
            }
        }

        /// <summary>
        /// Handle processing completed events
        /// </summary>
        private static void OnProcessingCompleted(object? sender, SeparationResult result)
        {
            Console.WriteLine($"\n✅ Completed in {result.ProcessingTime:mm\\:ss\\.ff}");
        }

        /// <summary>
        /// Handle error events
        /// </summary>
        private static void OnErrorOccurred(object? sender, Exception error)
        {
            Console.WriteLine($"\n❌ Error: {error.Message}");
        }

        #endregion
    }

    /// <summary>
    /// System information structure for optimization decisions
    /// </summary>
    public struct SystemInfo
    {
        public int CpuCores { get; set; }
        public long AvailableMemory { get; set; }
        public long EstimatedTotalRAM { get; set; }
        public bool Is64Bit { get; set; }
        public bool HasHighMemory { get; set; }
        public bool HasManyCores { get; set; }
    }
}


