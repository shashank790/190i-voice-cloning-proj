import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt

def get_audio_metrics(file_path):
    start_time = time.time()
    y, sr = librosa.load(file_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)
    size_kb = os.path.getsize(file_path) / 1024

    # RMS
    rms = np.sqrt(np.mean(y**2))

    # Peak
    peak = np.max(np.abs(y))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Spectral Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Spectral Rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {
        "time": elapsed_time,
        "size_kb": size_kb,
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff
    }

# Paths to audio files
initial_local_model = "comparison-audios/initial_local_model.wav"
updated_local_model = "comparison-audios/updated_local_model.wav"
elevenlabs_tts = "comparison-audios/elevenlabs_tts.wav"
azure_tts = "comparison-audios/azure_tts.wav"

# Get metrics separately for each file
metrics_initial = get_audio_metrics(initial_local_model)
metrics_updated = get_audio_metrics(updated_local_model)
metrics_elevenlabs = get_audio_metrics(elevenlabs_tts)
metrics_azure = get_audio_metrics(azure_tts)

# Print the results in a consistent, neatly aligned table
print("| Engine                                        | Time (s) | Size (KB) |   RMS   |  Peak  |   ZCR   | Centroid | Bandwidth | Rolloff |")
print("|-----------------------------------------------|----------|-----------|---------|--------|---------|----------|-----------|---------|")
print("| Initial Local Model                           | {time:<8.2f} | {size_kb:<9.1f} | {rms:<7.4f} | {peak:<6.4f} | {zcr:<7.4f} | {centroid:<8.1f} | {bandwidth:<9.1f} | {rolloff:<7.1f} |".format(**metrics_initial))
print("| Refined Local Model (w/ Augmentation & Preproc)| {time:<8.2f} | {size_kb:<9.1f} | {rms:<7.4f} | {peak:<6.4f} | {zcr:<7.4f} | {centroid:<8.1f} | {bandwidth:<9.1f} | {rolloff:<7.1f} |".format(**metrics_updated))
print("| ElevenLabs TTS                                | {time:<8.2f} | {size_kb:<9.1f} | {rms:<7.4f} | {peak:<6.4f} | {zcr:<7.4f} | {centroid:<8.1f} | {bandwidth:<9.1f} | {rolloff:<7.1f} |".format(**metrics_elevenlabs))
print("| Azure TTS                                      | {time:<8.2f} | {size_kb:<9.1f} | {rms:<7.4f} | {peak:<6.4f} | {zcr:<7.4f} | {centroid:<8.1f} | {bandwidth:<9.1f} | {rolloff:<7.1f} |".format(**metrics_azure))

# Prepare data
engines = ["Initial Local Model", "Refined Local Model", "ElevenLabs TTS", "Azure TTS"]

times = [metrics_initial["time"], metrics_updated["time"], metrics_elevenlabs["time"], metrics_azure["time"]]
sizes = [metrics_initial["size_kb"], metrics_updated["size_kb"], metrics_elevenlabs["size_kb"], metrics_azure["size_kb"]]
rms = [metrics_initial["rms"], metrics_updated["rms"], metrics_elevenlabs["rms"], metrics_azure["rms"]]
peak = [metrics_initial["peak"], metrics_updated["peak"], metrics_elevenlabs["peak"], metrics_azure["peak"]]
zcr = [metrics_initial["zcr"], metrics_updated["zcr"], metrics_elevenlabs["zcr"], metrics_azure["zcr"]]
centroid = [metrics_initial["centroid"], metrics_updated["centroid"], metrics_elevenlabs["centroid"], metrics_azure["centroid"]]
bandwidth = [metrics_initial["bandwidth"], metrics_updated["bandwidth"], metrics_elevenlabs["bandwidth"], metrics_azure["bandwidth"]]
rolloff = [metrics_initial["rolloff"], metrics_updated["rolloff"], metrics_elevenlabs["rolloff"], metrics_azure["rolloff"]]

# Create subplots
fig, axs = plt.subplots(4, 2, figsize=(12, 12))
fig.suptitle('Audio Metrics Comparison')

# Helper for line plots
def plot_metric(ax, y_values, title, ylabel):
    ax.plot(engines, y_values, marker='o')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, rotation=30, ha='right')

# Plot each metric
plot_metric(axs[0, 0], times, 'Processing Time', 'Time (s)')
plot_metric(axs[0, 1], sizes, 'File Size', 'Size (KB)')
plot_metric(axs[1, 0], rms, 'Root Mean Square (RMS)', 'RMS')
plot_metric(axs[1, 1], peak, 'Peak Amplitude', 'Peak')
plot_metric(axs[2, 0], zcr, 'Zero Crossing Rate (ZCR)', 'ZCR')
plot_metric(axs[2, 1], centroid, 'Spectral Centroid', 'Centroid (Hz)')
plot_metric(axs[3, 0], bandwidth, 'Spectral Bandwidth', 'Bandwidth (Hz)')
plot_metric(axs[3, 1], rolloff, 'Spectral Rolloff', 'Rolloff (Hz)')

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
plt.savefig("audio_metrics_comparison.png")
print("Line plots saved as: audio_metrics_comparison.png")