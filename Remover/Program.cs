using OwnSeparator.Core;

namespace OwnSeparator.BasicConsole
{
    /// <summary>
    /// Main program class for the OwnSeparator console application
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
            Console.WriteLine("OwnSeparator Audio Separation");
            Console.WriteLine("=========================");

            try
            {
                // Create service instance
                var service = AudioSeparationFactory.CreateBatchOptimized(
                    "models/OWN_INST_DEFAULT.ONNX", "output");

                // Subscribe to events
                service.ProgressChanged += (s, progress) =>
                    Console.WriteLine($"{progress.Status}: {progress.OverallProgress:F1}%");

                service.ProcessingCompleted += (s, result) =>
                    Console.WriteLine($"Completed: {result.ProcessingTime}");

                // Initialize and process
                Console.WriteLine("Initializing...");
                await service.InitializeAsync();

                Console.WriteLine("Starting processing...");
                var result = await service.SeparateAsync(@"D:\Sogorock\Ocam\2025\Unnep\Zorán -Az ünnep (cover)_audio.flac");

                Console.WriteLine($"Vocals file: {result.VocalsPath}");
                Console.WriteLine($"Instrumental file: {result.InstrumentalPath}");

                // Release resources
                service.Dispose();
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine($"File not found: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }

            Console.WriteLine("Press any key to exit...");
            Console.Read();
        }
    }
}