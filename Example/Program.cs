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
                //Create service optimized for mobile devices (faster processing)
                SeparationOptions options = new SeparationOptions
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

                ////Create service for batch processing (balanced settings)
                //SeparationOptions options = new SeparationOptions
                //{
                //    ModelPath = "models/OWN_INST_BEST.ONNX",
                //    OutputDirectory = "output",
                //    ChunkSizeSeconds = 15,
                //    Margin = 6144,
                //    NFft = 4096,
                //    DimT = 8,
                //    DimF = 2048,
                //    DisableNoiseReduction = true
                //};

                ////Create service optimized for desktop (better quality)
                //SeparationOptions options = new SeparationOptions
                //{
                //    ModelPath = "models/OWN_INST_BEST.ONNX",
                //    OutputDirectory = "output",
                //    ChunkSizeSeconds = 25, 
                //    Margin = 88200,        
                //    NFft = 8192,           
                //    DimT = 9,              
                //    DimF = 4096,
                //    DisableNoiseReduction = false
                //};

                // Create service instance
                var service = new AudioSeparationService( options);

                // Subscribe to events
                service.ProgressChanged += (s, progress) =>
                    Console.WriteLine($"{progress.Status}: {progress.OverallProgress:F1}%");

                service.ProcessingCompleted += (s, result) =>
                    Console.WriteLine($"Completed: {result.ProcessingTime}");

                // Initialize and process
                Console.WriteLine("Initializing...");
                await service.InitializeAsync();

                Console.WriteLine("Starting processing...");
                var result = await service.SeparateAsync(@"input/audio.mp3");

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
