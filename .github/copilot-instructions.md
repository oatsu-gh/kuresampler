# Copilot Instructions for kuresampler

## Project Overview

**kuresampler** is a UTAU engine that uses WORLD vocoder and neural network vocoders to achieve natural crossfading and high-quality audio output. It serves as both a resampler and wavtool for UTAU singing synthesis, targeting parametric voice synthesis with neural vocoder enhancement.

### Repository Summary
- **Type**: Audio processing/synthesis engine for UTAU
- **Language**: Python (main), C# (wrapper), Batch scripts (Windows build)
- **Size**: ~2,400 lines of code across 6 main Python modules
- **Target**: Windows with embedded Python 3.12.10
- **Dependencies**: NNSVS, PyTorch, WORLD vocoder, librosa, numpy

## Architecture & Project Layout

### Core Components
- **kuresampler.cs** (79 lines): C# wrapper that launches Python child processes (`kuresampler_child.bat`)
- **kuresampler.py** (474 lines): Main render engine with `NeuralNetworkRender` class
- **resampler.py** (406 lines): Core resampler classes (`WorldFeatureResamp`, `NeuralNetworkResamp`)
- **wavtool.py** (466 lines): Audio crossfading and mixing (`WorldFeatureWavTool`)
- **convert.py** (328 lines): Audio format conversions (WAV ↔ WORLD ↔ NNSVS features)
- **util.py** (214 lines): Utilities for logging, device detection, model loading
- **test.py** (421 lines): Test functions for development and validation

### Directory Structure
```
/
├── .github/                     # GitHub configuration
├── models/                      # Pre-trained neural vocoder models
│   ├── README.md               # Model documentation and licenses
│   └── usfGAN_*/               # uSFGAN and other vocoder models
├── test/                       # Test data and UST files
├── data/                       # Additional data files
├── wrapper/Resampler/          # Fast wrapper variants
├── requirements.txt            # Python dependencies (git+custom forks)
├── pyproject.toml             # Ruff linting configuration
└── *.bat                      # Windows batch scripts for setup/compilation
```

### Configuration Files
- **pyproject.toml**: Ruff linter config (line-length=99, Python 3.12 target)
- **requirements.txt**: Git-based dependencies from custom NNSVS forks
- **.gitignore**: Excludes .exe, .wav, .npz, cache dirs

## Build & Environment Setup

### Prerequisites
- **Windows environment** (primary target platform)
- **Python 3.12.10 embedded** (production) or regular Python 3.12+ (development)
- **.NET Framework 4.0** (for C# wrapper compilation)
- **CUDA GPU recommended** (fallback to CPU supported)

### Development Environment Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch torchaudio torchvision  # Install manually per environment
   # Note: requirements.txt contains git+https:// dependencies that may take 5-10 minutes to install
   ```

2. **For embedded Python environment** (Windows production):
   ```bash
   # Setup embedded Python with additional DLLs and headers
   # Copy from full Python installation:
   # - python/include/ → python-embeddable/include/
   # - python/libs/ → python-embeddable/libs/
   # - python/tcl/ → python-embeddable/tcl/
   # - python/Lib/tkinter/ → python-embeddable/tkinter/
   # - Various DLLs: _tkinter.pyd, tcl86t.dll, tk86t.dll, zlib1.dll
   
   python-3.12.10-embed-amd64\python.exe -m pip install -r requirements.txt --no-warn-script-location
   ```

3. **PyTorch installation** (use light-the-torch for optimal GPU support):
   ```bash
   python -m pip install --upgrade light-the-torch
   python -m light_the_torch install torch torchaudio torchvision
   # Or run: reinstall_torch.bat
   ```

### Build Commands

1. **Compile C# wrapper**:
   ```bash
   # Windows only - requires .NET Framework
   _compile.bat  # Compiles kuresampler.cs → kuresampler.exe
   ```

2. **Linting**:
   ```bash
   ruff check --config pyproject.toml .  # Many style warnings expected (D212, EXE001, etc.)
   ruff format --config pyproject.toml .
   ```

3. **Clean cache**:
   ```bash
   clean_pycache.bat  # Windows
   # Or manually: find . -name "*.pyc" -delete; find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

## Testing & Validation

### Test Execution
```bash
python test.py  # Runs comprehensive test suite including:
                # - WORLD feature conversion tests
                # - Neural vocoder model validation
                # - Performance benchmarks
                # - Full resampler+wavtool pipeline
                # Note: Requires full dependency installation including utaupy, PyRwu, etc.
```

### Test Components
- **test_convert()**: WAV ↔ WORLD ↔ NNSVS feature conversion
- **test_vocoder_model()**: Neural vocoder model loading and inference
- **test_performance()**: Timing benchmarks for bottleneck functions
- **test_resampler_and_wavtool()**: Full UTAU pipeline simulation

### Manual Testing
```bash
# Test resampler functionality
python resampler.py [args] --model_dir ./models/usfGAN_EnunuKodoku_0826/

# Test via batch wrapper
kuresampler_child.bat [resampler_args]
```

## Common Issues & Workarounds

### Dependency Issues
- **Missing docopt**: Copy from full Python installation to embedded Python
- **Missing tcl/tk**: Copy tcl/, tkinter/, and related DLLs for uSFGAN support
- **pysptk compilation**: Requires include/ and libs/ directories from full Python

### Performance Notes
- **Resampling**: SOXR types ['soxr_vhq', 'soxr_hq', 'kaiser_best'] - balance quality vs speed
- **Large models**: uSFGAN models require significant GPU memory
- **f0 estimation**: Harvest algorithm is slow; consider using .frq files when available

### Known Limitations
- **Windows-focused**: Batch scripts and embedded Python target Windows
- **Memory intensive**: Neural vocoders require 2GB+ GPU memory
- **Model dependency**: Requires pre-trained NNSVS vocoder models in models/ directory

## Development Notes

### TODO Items (from codebase analysis)
- WorldFeatureResamp integration into NeuralNetworkRender
- Volume normalization for WORLD features
- Automatic p_list sorting in wavtool
- .frq file support for faster f0 estimation
- NPZ volume adjustment integration

### Key Design Patterns
- **Wrapper pattern**: C# exe → batch → Python for UTAU compatibility
- **Feature pipelines**: WAV → WORLD → NNSVS → Neural Vocoder → WAV
- **Caching**: NPZ files store intermediate WORLD/NNSVS features
- **Modular vocoders**: Support for uSFGAN, ParallelWaveGAN, SiFiGAN

### Important Constants & Defaults
- **Sample rates**: 44.1kHz (input) → 48kHz (processing)
- **Frame period**: 5.0ms for WORLD analysis
- **F0 range**: 150-700Hz (defined in test.py)
- **Model paths**: `./models/usfGAN_Namineritsu_4130/` (default), `./models/usfGAN_EnunuKodoku_0826/` (alternative)

### File Extensions & Formats
- **.wav**: Audio input/output
- **.npz**: Cached WORLD/NNSVS features (numpy arrays)
- **.ust**: UTAU project files
- **.frq**: Frequency analysis cache (not yet implemented)
- **.exe**: Compiled C# wrapper

## Trust These Instructions

These instructions are comprehensive and tested. Only perform additional searches if:
- Instructions are incomplete for your specific task
- You encounter errors not covered in the "Common Issues" section
- You need to modify core architecture or add new vocoder support

The codebase is well-structured with clear separation between audio processing, neural inference, and UTAU integration layers. Most modifications should focus on the Python modules rather than the C#/batch wrapper system.