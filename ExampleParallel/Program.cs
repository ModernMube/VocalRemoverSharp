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

            string audioFilePath = @"input/audio.mp3";
            string modelFilePath = @"models/OWN_INST_BEST.ONNX";
            string outputDirectory = @"output";

            try
            {
                (var service, var parallelOptions) = AudioSeparationFactory.CreateSystemOptimized(modelFilePath, outputDirectory);

                service.ProgressChanged += OnProgressChanged;
                service.ProcessingStarted += OnProcessingStarted;
                service.ProcessingCompleted += OnProcessingCompleted;
                service.ErrorOccurred += OnErrorOccurred;

                Console.WriteLine("\nInitializing parallel processing...");
                var initStopwatch = Stopwatch.StartNew();

                await service.InitializeParallelAsync(parallelOptions);

                initStopwatch.Stop();
                Console.WriteLine($"Initialization completed in {initStopwatch.ElapsedMilliseconds}ms");

                await ProcessSingleFile(service, audioFilePath);

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
}


