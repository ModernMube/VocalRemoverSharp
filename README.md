# OwnVocalRemover

A high-performance audio source separation library for .NET that uses ONNX models to separate vocals and instrumental tracks from mixed audio files. 
I've been looking for a good vocal and music separation code in C# for a very long time that provides decent quality. 
Unfortunately, I could only find such code in Python, so I decided to create a pure Csharp vocal separator that would deliver the quality created by the Python code!

## Features

- **ONNX Model Support**: Works with pre-trained ONNX models for audio separation
- **GPU Acceleration**: Automatic CUDA support with CPU fallback
- **Parallel Processing**: Multi-threaded chunk processing with session pooling
- **Memory Management**: Adaptive memory pressure monitoring
- **Chunked Processing**: Handles large files by processing in configurable chunks
- **Noise Reduction**: Optional denoising for improved separation quality
- **Batch Processing**: Process multiple files efficiently
- **Progress Tracking**: Real-time progress reporting with events
- **Auto-Configuration**: Automatically detects model parameters from ONNX metadata

## Dependencies

- `MathNet.Numerics` - FFT operations
- `Microsoft.ML.OnnxRuntime` - ONNX model inference
- `Ownaudio` - Audio I/O operations
- `Microsoft.Extensions.ObjectPool` - Session pooling for parallel processing

## Support My Work

If you find this project helpful, consider buying me a coffee!

<a href="https://www.buymeacoffee.com/ModernMube" 
    target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" 
    alt="Buy Me A Coffee" 
    style="height: 60px !important;width: 217px !important;" >
 </a>

## Quick Start

### Basic Usage (Traditional Mode)

```csharp
// Basic usage with included model
var service = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_DEFAULT.onnx");
await service.InitializeAsync();

var result = await service.SeparateAsync("input_song.wav");
Console.WriteLine($"Vocals: {result.VocalsPath}");
Console.WriteLine($"Instrumental: {result.InstrumentalPath}");

service.Dispose();
```

### Parallel Processing Mode

```csharp
// Parallel processing for faster performance
var service = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_DEFAULT.onnx");

var parallelOptions = new ParallelProcessingOptions
{
    MaxDegreeOfParallelism = 4,
    SessionPoolSize = 3,
    EnableMemoryPressureMonitoring = true
};

await service.InitializeParallelAsync(parallelOptions);
var result = await service.SeparateAsync("input_song.wav");

service.Dispose();
```

## Configuration Options

### SeparationOptions

- **ModelPath**: Path to ONNX model file
- **OutputDirectory**: Output directory for separated files
- **DisableNoiseReduction**: Disable denoising (default: false)
- **Margin**: Overlap margin for chunks (default: 44100 samples)
- **ChunkSizeSeconds**: Chunk duration in seconds (0 = process entire file)
- **NFft**: FFT size (default: 6144)
- **DimT**: Temporal dimension parameter (default: 8)
- **DimF**: Frequency dimension parameter (default: 2048)

### ParallelProcessingOptions

- **MaxDegreeOfParallelism**: Maximum concurrent chunks (0 = auto-detect)
- **SessionPoolSize**: Number of ONNX sessions in pool (0 = auto-detect)
- **EnableMemoryPressureMonitoring**: Monitor memory usage (default: true)
- **MemoryPressureThreshold**: Memory threshold in bytes (default: 2GB)
- **ChunkQueueCapacity**: Queue capacity for chunks (default: 10)

## Usage Examples

### Custom Configuration with Parallel Processing

```csharp
var options = new SeparationOptions
{
    ModelPath = "models/my_model.onnx",
    OutputDirectory = "output",
    ChunkSizeSeconds = 20,
    DisableNoiseReduction = false
};

var parallelOptions = new ParallelProcessingOptions
{
    MaxDegreeOfParallelism = 6,
    SessionPoolSize = 4,
    EnableMemoryPressureMonitoring = true,
    MemoryPressureThreshold = 3_000_000_000 // 3GB
};

var service = new AudioSeparationService(options);
await service.InitializeParallelAsync(parallelOptions);
```

### System-Optimized Configuration

```csharp
// Automatically configure based on system capabilitiesvar 
var (service, parallelOptions) = AudioSeparationFactory.CreateSystemOptimized("models/my_model.onnx", "output");
await service.InitializeParallelAsync(parallelOptions);
```

### Progress Monitoring

```csharp
service.ProgressChanged += (sender, progress) =>
{
    Console.WriteLine($"Progress: {progress.OverallProgress:F1}% - {progress.Status}");
    Console.WriteLine($"Chunks: {progress.ProcessedChunks}/{progress.TotalChunks}");
};

service.ProcessingStarted += (sender, file) =>
{
    Console.WriteLine($"Started processing: {file}");
};

service.ProcessingCompleted += (sender, result) =>
{
    Console.WriteLine($"Completed in {result.ProcessingTime}");
};
```

### Batch Processing

```csharp
var files = new[] { "song1.wav", "song2.wav", "song3.wav" };
var results = await service.SeparateMultipleAsync(files);

foreach (var result in results)
{
    Console.WriteLine($"Processed: {result.VocalsPath}");
}
```

## Pre-configured Factory Methods

### Mobile Optimized (Faster)
```csharp
var service = AudioSeparationFactory.CreateMobileOptimized(
    "models/model.onnx", 
    "output", 
    disableNoiseReduction: true
);
await service.InitializeAsync(); // Traditional mode for mobile
```

### Desktop Optimized (Better Quality)
```csharp
var service = AudioSeparationFactory.CreateDesktopOptimized(
    "models/model.onnx", 
    "output"
);
await service.InitializeParallelAsync(); // Parallel mode for desktop
```

### System-Optimized with Parallel Processing
```csharp
var (service, parallelOptions) = AudioSeparationFactory.CreateSystemOptimized(
    "models/model.onnx", 
    "output",
    systemCores: Environment.ProcessorCount,
    availableMemoryGB: 16.0
);
await service.InitializeParallelAsync(parallelOptions);
```

### Choosing the Right Model

**For general use**: Start with `OWN_INST_DEFAULT.onnx`
```csharp
var service = AudioSeparationFactory.CreateBatchOptimized("models/OWN_INST_DEFAULT.onnx", "output");
```

**For best quality**: Use `OWN_INST_BEST.onnx` with desktop settings
```csharp
var service = AudioSeparationFactory.CreateDesktopOptimized("models/OWN_INST_BEST.onnx", "output");
```

**For karaoke creation**: Use `OWN_KAR.onnx`
```csharp
var service = AudioSeparationExtensions.CreateDefaultService("models/OWN_KAR.onnx");
```

**For custom MDXNET models**: Any compatible model works
```csharp
var service = AudioSeparationExtensions.CreateDefaultService("models/custom_mdxnet.onnx");
```

## Processing Modes

### Traditional Mode
- Single-threaded processing
- Lower memory usage
- Suitable for mobile/low-end devices
- Initialize with `InitializeAsync()`

### Parallel Processing Mode
- Multi-threaded chunk processing
- Higher performance on multi-core systems
- Session pooling for better resource utilization
- Memory pressure monitoring
- Initialize with `InitializeParallelAsync()`

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)

## Output Files

The service generates two files per input:
- `{filename}_vocals.wav` - Extracted vocals
- `{filename}_music.wav` - Instrumental track

## Error Handling

```csharp
try
{
    var result = await service.SeparateAsync("input.wav");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Service error: {ex.Message}");
}
catch (AggregateException ex) when (ex.InnerExceptions.Any())
{
    Console.WriteLine("Parallel processing errors occurred:");
    foreach (var innerEx in ex.InnerExceptions)
    {
        Console.WriteLine($"- {innerEx.Message}");
    }
}
```

## Performance Tips

1. **Processing Mode**: Use parallel processing on multi-core systems
2. **GPU Acceleration**: Ensure CUDA is available for faster processing
3. **Chunk Size**: Adjust `ChunkSizeSeconds` based on available memory
4. **Session Pool**: Increase `SessionPoolSize` for better parallel performance
5. **Memory Management**: Enable memory pressure monitoring for large files
6. **Noise Reduction**: Disable for faster processing in batch scenarios

## Memory Management

The parallel processing mode includes adaptive memory management:

- **Memory Pressure Monitoring**: Automatically detects high memory usage
- **Garbage Collection**: Forces GC under memory pressure
- **Throttling**: Reduces parallelism when memory is constrained
- **Session Pooling**: Efficient reuse of ONNX sessions

## Statistics and Analysis

The `SeparationResult` includes audio statistics:

```csharp
var stats = result.Statistics;
Console.WriteLine($"Vocals RMS: {stats.VocalsRMS:F4}");
Console.WriteLine($"Instrumental RMS: {stats.InstrumentalRMS:F4}");
Console.WriteLine($"Sample Rate: {stats.SampleRate} Hz");
Console.WriteLine($"Processing Time: {result.ProcessingTime}");
```

## Included Models

The library comes with three pre-trained models:

### OWN_INST_DEFAULT.onnx
- **Type**: Basic instrumental separation
- **Quality**: Good baseline performance
- **Use case**: General purpose separation, fastest processing
- **Output**: Clean vocals and instrumental tracks

### OWN_INST_BEST.onnx  
- **Type**: High-quality instrumental separation
- **Quality**: Superior separation accuracy
- **Use case**: When quality is more important than speed
- **Output**: High-fidelity vocals and instrumental tracks

### OWN_KAR.onnx
- **Type**: Karaoke model (lead vocal removal)
- **Quality**: Specialized for karaoke creation
- **Use case**: Remove lead vocals while preserving backing vocals
- **Output**: Lead vocals and music with backing vocals intact

## Model Usage Examples

```csharp
// Using the default model
var defaultService = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_DEFAULT.onnx");

// Using the best quality model with parallel processing
var bestService = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_BEST.onnx");
await bestService.InitializeParallelAsync();

// Using the karaoke model
var karaokeService = AudioSeparationExtensions.CreateDefaultService("models/OWN_KAR.onnx");
```

## MDXNET Model Support

The library is fully compatible with any MDXNET model:

```csharp
// Using custom MDXNET model with parallel processing
var mdxService = AudioSeparationExtensions.CreateDefaultService("models/my_mdxnet_model.onnx");
await mdxService.InitializeParallelAsync(); // Auto-detects model parameters
```

## Model Requirements

ONNX models should:
- Accept input shape: `[batch, 4, frequency, time]`
- Output same shape as input
- Support 44.1kHz stereo audio
- Use STFT-based processing
- Be compatible with MDXNET architecture

## Thread Safety

- The `AudioSeparationService` is **not thread-safe** for concurrent operations on the same instance
- Parallel processing is handled internally and is thread-safe
- Create separate instances for concurrent processing of different files
- Session pooling ensures safe concurrent access to ONNX models

## Best Practices

### For Single Files
```csharp
using var service = AudioSeparationExtensions.CreateDefaultService("models/model.onnx");
await service.InitializeParallelAsync();
var result = await service.SeparateAsync("song.wav");
```

### For Batch Processing
```csharp
var service = AudioSeparationFactory.CreateBatchOptimized("models/model.onnx", "output");
await service.InitializeParallelAsync();

var files = Directory.GetFiles("input", "*.wav");
var results = await service.SeparateMultipleAsync(files);

service.Dispose();
```

### For System-Specific Optimization
```csharp
var (service, options) = AudioSeparationFactory.CreateSystemOptimized(
    "models/model.onnx", 
    "output",
    Environment.ProcessorCount,
    GC.GetTotalMemory(false) / (1024.0 * 1024.0 * 1024.0) // Available memory in GB
);
await service.InitializeParallelAsync(options);
```

## Disposal

Always dispose the service to free ONNX resources and session pools:

```csharp
using var service = new AudioSeparationService(options);
await service.InitializeParallelAsync();
// Use service...
// Automatically disposed, including session pool cleanup
```
