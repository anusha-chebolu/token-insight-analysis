import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths to your existing CSVs
csv_paths = {
    "DeepSeek 7B": "../results/deepseek_surprisal.csv",
    "DeepSeek R1 Distill 1.5B": "../results/deepseek_qwen1.5b_surprisal.csv",
    "Qwen 2-1.5B": "../results/qwen2-1.5b_surprisal.csv",
    "Qwen 2.5-7B Instruct": "../results/qwen2-5-7b_surprisal.csv"
}

# Color scheme per model
model_colors = {
    "DeepSeek 7B": "#8E44AD",
    "DeepSeek R1 Distill 1.5B": "#4B3C8E",
    "Qwen 2-1.5B": "#228B22",
    "Qwen 2.5-7B Instruct": "#4CAF50"
}

# Load, clean, and label data
dfs = []
for model_name, path in csv_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Word_ID"].isin(range(1, 9))]  # Only Word_ID 1 to 8
        df["Model"] = model_name
        dfs.append(df)
    else:
        print(f"Warning: File not found -> {path}")

# Exit early if nothing was loaded
if not dfs:
    raise ValueError("No CSVs found. Please check file paths.")

# Combine and group
combined_df = pd.concat(dfs, ignore_index=True)
avg_df = combined_df.groupby(["Model", "Word_ID"])["Surprisal value"].mean().reset_index()

# Plot 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

for idx, (model_name, group) in enumerate(avg_df.groupby("Model")):
    ax = axes[idx]
    ax.bar(
        group["Word_ID"],
        group["Surprisal value"],
        color=model_colors.get(model_name, "gray"),
        width=0.6
    )
    ax.set_title(model_name, fontsize=12)
    ax.set_xlabel("Word Position (Word_ID)")
    if idx % 2 == 0:
        ax.set_ylabel("Average Surprisal")
    ax.set_xticks(group["Word_ID"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Super title and layout
fig.suptitle("Average Word Surprisal Comparison Across Models (Word_IDs 1–8)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save plot
output_path = "../results/model_surprisal_comparison_4grid.png"
plt.savefig(output_path)
plt.close()

print(f"Combined plot saved to: {output_path}")
