import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import warnings
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Fix plot display issue (Windows)
matplotlib.use('TkAgg')

# ==========================
# 1. LOAD DATA
# ==========================
file_path = r"C:\Users\rayav\Downloads\Final_Resultsnew.csv"

if not os.path.exists(file_path):
    print(" File not found")
    exit()

data = pd.read_csv(file_path)
print(" Data Loaded Successfully")

# ==========================
# 2. RENAME Q1–Q42
# ==========================
new_columns = {}
for col in data.columns:
    match = re.match(r"(\d+)\)", str(col).strip())
    if match:
        new_columns[col] = "Q" + match.group(1)

data.rename(columns=new_columns, inplace=True)
data = data.loc[:, ~data.columns.str.contains("Unnamed")]

# ==========================
# 3. DEFINE ITEMS
# ==========================
dep_items = [3,5,10,13,16,17,21,24,26,31,34,37,38,42]
anx_items = [2,4,7,9,15,19,20,23,25,28,30,36,40,41]
str_items = [1,6,8,11,12,14,18,22,27,29,32,33,35,39]

dep_cols = ["Q"+str(i) for i in dep_items]
anx_cols = ["Q"+str(i) for i in anx_items]
str_cols = ["Q"+str(i) for i in str_items]

data[dep_cols + anx_cols + str_cols] = data[dep_cols + anx_cols + str_cols] \
    .apply(pd.to_numeric, errors='coerce').fillna(0)

# ==========================
# 4. CALCULATE SCORES
# ==========================
data["Depression"] = data[dep_cols].sum(axis=1) * 2
data["Anxiety"]    = data[anx_cols].sum(axis=1) * 2
data["Stress"]     = data[str_cols].sum(axis=1) * 2

# ==========================
# 5. SEVERITY LABELS
# ==========================
def dep_sev(x):
    if x <= 9:  return "Normal"
    elif x <= 13: return "Mild"
    elif x <= 20: return "Moderate"
    elif x <= 27: return "Severe"
    else:         return "Extremely Severe"

def anx_sev(x):
    if x <= 7:  return "Normal"
    elif x <= 9:  return "Mild"
    elif x <= 14: return "Moderate"
    elif x <= 19: return "Severe"
    else:         return "Extremely Severe"

def str_sev(x):
    if x <= 14: return "Normal"
    elif x <= 18: return "Mild"
    elif x <= 25: return "Moderate"
    elif x <= 33: return "Severe"
    else:         return "Extremely Severe"

data["Depression_Severity"] = data["Depression"].apply(dep_sev)
data["Anxiety_Severity"]    = data["Anxiety"].apply(anx_sev)
data["Stress_Severity"]     = data["Stress"].apply(str_sev)

# ==========================
# 6. SAVE OUTPUT
# ==========================
output_dir = r"C:\Users\rayav\Downloads\DASS_Output"
os.makedirs(output_dir, exist_ok=True)

data.to_csv(os.path.join(output_dir, "Final_Results.csv"), index=False)

# ==========================
# 7. VISUALIZATIONS
# ==========================

# --- BOXPLOT ---
data_long = data.melt(
    value_vars=["Depression", "Anxiety", "Stress"],
    var_name="Subscale",
    value_name="Score"
)

plt.figure(figsize=(8, 6))
sns.boxplot(x="Subscale", y="Score", data=data_long)
plt.title("DASS Score Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Boxplot.png"))
plt.show(block=True)

# --- BARPLOT ---
severity_data = data.melt(
    value_vars=["Depression_Severity", "Anxiety_Severity", "Stress_Severity"],
    var_name="Subscale",
    value_name="Severity"
)
severity_data["Subscale"] = severity_data["Subscale"].str.replace("_Severity", "")

plt.figure(figsize=(10, 6))
sns.countplot(x="Subscale", hue="Severity", data=severity_data)
plt.title("Severity Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Barplot.png"))
plt.show(block=True)

# --- SCATTER PLOT (Fixed: numeric size values removed from legend) ---
data["Dep_jitter"] = data["Depression"] + np.random.uniform(-1, 1, len(data))
data["Anx_jitter"] = data["Anxiety"]    + np.random.uniform(-1, 1, len(data))

plt.figure(figsize=(9, 7))
sns.scatterplot(
    x="Dep_jitter",
    y="Anx_jitter",
    hue="Stress_Severity",
    palette="coolwarm",
    size="Stress",
    sizes=(40, 200),
    alpha=0.6,
    edgecolor="black",
    data=data
)

plt.title("Depression vs Anxiety (Clear Scatter)")
plt.xlabel("Depression Score")
plt.ylabel("Anxiety Score")
plt.grid(True, linestyle="--", alpha=0.5)

# Keep only Stress_Severity category labels — remove numeric size entries
handles, leg_labels = plt.gca().get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(handles, leg_labels) if not l.replace('.', '').isdigit()]
plt.legend(
    [h for h, l in filtered],
    [l for h, l in filtered],
    title="Stress Severity",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Scatter_Clear.png"))
plt.show(block=True)

# ==========================
# 8. MACHINE LEARNING
# ==========================
print("\n TRAINING MODEL...")

def get_label(x):
    if x <= 9:  return 0
    elif x <= 13: return 1
    elif x <= 20: return 2
    elif x <= 27: return 3
    else:         return 4

data["Dep_Label"] = data["Depression"].apply(get_label)

feature_cols = [f'Q{i}' for i in range(1, 43) if f'Q{i}' in data.columns]

X = data[feature_cols]
y = data["Dep_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================
# 9. RESULTS
# ==========================
print("\n Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Confusion Matrix — clean printed table
print("\n Confusion Matrix:\n")
class_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
cm_df.index.name   = "Actual"
cm_df.columns.name = "Predicted"
print(cm_df)

# Per-class accuracy summary (clean, no warnings)
print("\n Per-Class Summary:\n")
print(f"{'Class':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("-" * 50)
for i, label in enumerate(class_labels):
    total   = cm[i].sum()
    correct = cm[i][i]
    acc     = f"{round(correct / total * 100, 1)}%" if total > 0 else "N/A"
    print(f"{label:<20} {correct:>8} {total:>8} {acc:>10}")

# ==========================
# 10. SAVE OUTPUTS
# ==========================
cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

importances = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
importances.to_csv(os.path.join(output_dir, "feature_importance.csv"))

# ==========================
# DONE
# ==========================
os.startfile(output_dir)

print("\n PROJECT COMPLETED SUCCESSFULLY!")