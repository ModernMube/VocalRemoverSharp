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

            string audioFilePath = @"path/to/audio.mp3";
            string outputDirectory = @"output";

            try
            {
                (var service, var ParallelOptions) = AudioSeparationFactory.CreateSystemOptimized(InternalModel.Best, outputDirectory);

                service.ProgressChanged += (s, progress) =>
                    Console.WriteLine($"{progress.Status}: {progress.OverallProgress:F1}%");

                service.ProcessingCompleted += (s, result) =>
                    Console.WriteLine($"Completed: {result.ProcessingTime}");

                Console.WriteLine("Initializing...");
                await service.InitializeAsync();

                Console.WriteLine("Starting processing...");
                var result = await service.SeparateAsync(audioFilePath);

                Console.WriteLine($"Vocals file: {result.VocalsPath}");
                Console.WriteLine($"Instrumental file: {result.InstrumentalPath}");

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
