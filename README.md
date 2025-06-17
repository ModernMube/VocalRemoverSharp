# OwnVocalRemover

A high-performance audio source separation library for .NET that uses ONNX models to separate vocals and instrumental tracks from mixed audio files.
I've been looking for a good vocal and music separator code in C# for a very long time. Unfortunately, I could only find one in Python, so I decided to create a pure Csharp vocal separator based on the Python code I thought was the best!

## Features

- **ONNX Model Support**: Works with pre-trained ONNX models for audio separation
- **GPU Acceleration**: Automatic CUDA support with CPU fallback
- **Chunked Processing**: Handles large files by processing in configurable chunks
- **Noise Reduction**: Optional denoising for improved separation quality
- **Batch Processing**: Process multiple files efficiently
- **Progress Tracking**: Real-time progress reporting with events
- **Auto-Configuration**: Automatically detects model parameters from ONNX metadata

## Dependencies

- `MathNet.Numerics` - FFT operations
- `Microsoft.ML.OnnxRuntime` - ONNX model inference
- `Ownaudio` - Audio I/O operations

## Support My Work

If you find this project helpful, consider buying me a coffee!

<a href="https://www.buymeacoffee.com/ModernMube" 
    target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" 
    alt="Buy Me A Coffee" 
    style="height: 60px !important;width: 217px !important;" >
 </a>

## Quick Start

```csharp
// Basic usage with included model
var service = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_DEFAULT.onnx");
await service.InitializeAsync();

var result = await service.SeparateAsync("input_song.wav");
Console.WriteLine($"Vocals: {result.VocalsPath}");
Console.WriteLine($"Instrumental: {result.InstrumentalPath}");

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

## Usage Examples

### Custom Configuration

```csharp
var options = new SeparationOptions
{
    ModelPath = "models/my_model.onnx",
    OutputDirectory = "output",
    ChunkSizeSeconds = 20,
    DisableNoiseReduction = false
};

var service = new AudioSeparationService(options);
await service.InitializeAsync();
```

### Progress Monitoring

```csharp
service.ProgressChanged += (sender, progress) =>
{
    Console.WriteLine($"Progress: {progress.OverallProgress:F1}% - {progress.Status}");
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
```

### Desktop Optimized (Better Quality)
```csharp
var service = AudioSeparationFactory.CreateDesktopOptimized(
    "models/model.onnx", 
    "output"
);
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
```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster processing
2. **Chunk Size**: Adjust `ChunkSizeSeconds` based on available memory
3. **Noise Reduction**: Disable for faster processing in batch scenarios
4. **Memory**: Larger chunks require more memory but may improve quality

## Statistics and Analysis

The `SeparationResult` includes audio statistics:

```csharp
var stats = result.Statistics;
Console.WriteLine($"Vocals RMS: {stats.VocalsRMS:F4}");
Console.WriteLine($"Instrumental RMS: {stats.InstrumentalRMS:F4}");
Console.WriteLine($"Sample Rate: {stats.SampleRate} Hz");
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

// Using the best quality model
var bestService = AudioSeparationExtensions.CreateDefaultService("models/OWN_INST_BEST.onnx");

// Using the karaoke model
var karaokeService = AudioSeparationExtensions.CreateDefaultService("models/OWN_KAR.onnx");
```

## MDXNET Model Support

The library is fully compatible with any MDXNET model:

```csharp
// Using custom MDXNET model
var mdxService = AudioSeparationExtensions.CreateDefaultService("models/my_mdxnet_model.onnx");
await mdxService.InitializeAsync(); // Auto-detects model parameters
```

## Model Requirements

ONNX models should:
- Accept input shape: `[batch, 4, frequency, time]`
- Output same shape as input
- Support 44.1kHz stereo audio
- Use STFT-based processing
- Be compatible with MDXNET architecture

## Thread Safety

The `AudioSeparationService` is **not thread-safe**. Create separate instances for concurrent processing or use proper synchronization.

## Disposal

Always dispose the service to free ONNX resources:

```csharp
using var service = new AudioSeparationService(options);
// Use service...
// Automatically disposed
```
