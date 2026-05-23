#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


tree = pd.read_csv(r"C:\Users\14S11959\AppData\Roaming\Microsoft\Windows\Network Shortcuts\heart_failure_readmission_dataset.csv")


# In[3]:


# Step 3: View initial rows
tree.head()


# In[6]:


null_summary = tree.isnull().sum()
print(null_summary)


# In[7]:


tree = tree.drop(columns=['patient_id'])


# In[8]:


tree[['sodium','creatinine']] = tree[['sodium','creatinine']].apply(pd.to_datetime, errors='coerce')
tree['calculation'] = (tree['creatinine'] - tree['sodium']).dt.days


# In[9]:


tree.head()


# In[10]:


tree = tree.drop(columns=['sodium', 'creatinine'])


# In[11]:


tree.head()


# In[12]:


tree = tree.drop(columns=['systolic_bp', 'heart_rate'])


# In[13]:


tree.head()


# In[14]:


gender_map = {'Male': 0, 'M': 0, 'male': 0, 
              'Female': 1, 'F': 1, 'female': 1}
tree['gender'] = tree['gender'].map(gender_map)


# In[15]:


tree.head()


# In[16]:


tree['bmi'].nunique()
tree['bmi'].unique()


# In[19]:


tree.head()


# In[20]:


print(tree.columns.tolist())


# In[21]:


print(tree['income_level'].unique())
print(tree['income_level'].str.len().unique())  # length of strings


# In[23]:


from sklearn.preprocessing import LabelEncoder

# Initialize encoder
le = LabelEncoder()

# Fit and transform
tree['income level'] = le.fit_transform(tree['income_level'])

# Show mapping
print(dict(zip(le.classes_, le.transform(le.classes_))))


# In[24]:


tree.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encoding the column
tree['income level'] = le.fit_transform(tree['income_level'])

# Dropping the original column
tree.drop(columns=['income_level'], inplace=True)

print(tree.head())


# In[32]:


negatives = (tree < 1).any()

print("Columns with negative values:")
print(negatives[negatives == True])


# In[39]:


print("Number of negative adherence score:", (tree['adherence_score'] < 0).sum())


# In[41]:


# 1. Save the negative Billing Amount records separately
refund_records = tree[tree['adherence_score'] < 0].copy()

# Optional: save to CSV for future use
refund_records.to_csv('heart_failure_readmission_dataset.csv', index=False)

# 2. Create a new dataset without negative Billing Amounts for modeling
tree_clean = tree[tree['adherence_score'] >= 0].copy()


# 3. Verify
print("Original dataset size:", tree.shape[0])
print("Refund records size:", refund_records.shape[0])
print("Clean dataset size:", tree_clean.shape[0])


# In[42]:


tree_clean.head()


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr = tree_clean.corr()

# Correlation with target
print(corr['adherence_score'].sort_values(ascending=False))

# Optional: visualize entire correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# In[52]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encoding the column
tree['adherence_score'] = le.fit_transform(tree['adherence_score'])

# Dropping the original column
tree.drop(columns=['adherence_score'], inplace=True)

print(tree.head())


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(tree_clean['age'], bins=20, kde=True)
plt.title('age Distribution')
plt.xlabel('age')
plt.ylabel('Count')
plt.show()

# Optional: descriptive stats
print(tree_clean['age'].describe())


# In[54]:


tree_clean['age_group'] = pd.cut(
    tree_clean['age'],
    bins=[0, 18, 35, 52, 68, 100],
    labels=False
)

# Check distribution in each bucket
print(tree_clean['age_group'].value_counts().sort_index())


# In[55]:


tree_clean.head()


# In[66]:


# Separate numeric and categorical columns
numeric_cols = tree.select_dtypes(include=[np.number]).columns.tolist()
if 'readmitted_30d' in numeric_cols:
    numeric_cols.remove('readmitted_30d') # Keep target isolated
   
categorical_cols = tree.select_dtypes(exclude=[np.number]).columns.tolist()

# 1. Fill numeric null values with the median
for col in numeric_cols:
    tree[col] = tree[col].fillna(tree[col].median())

# 2. Fill categorical null values with the mode
for col in categorical_cols:
    tree[col] = tree[col].fillna(tree[col].mode()[0])

print("Null values remaining:", tree.isnull().sum().sum())


# In[71]:



# Cap outliers to prevent them from distorting KNN calculations
for col in numeric_cols:
    Q1 = tree[col].quantile(0.25)
    Q3 = tree[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
   
    # Winsorize/Cap the extreme values
    tree[col] = np.where(tree[col] < lower_bound, lower_bound, tree[col])
    tree[col] = np.where(tree[col] > upper_bound, upper_bound, tree[col])

print("Outliers capped successfully across numerical columns.")


# In[72]:


from sklearn.model_selection import train_test_split

# Encode categorical variables into numeric dummy columns
tree_encoded = pd.get_dummies(tree, columns=categorical_cols, drop_first=True)

# Separate features (X) and target label (y)
X = tree_encoded.drop(columns=['readmitted_30d'])
y = tree_encoded['readmitted_30d']

# Split into 80% Training and 20% Testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[73]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Create copies specific to KNN to preserve pure data structures
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert back to dataframes/arrays for structural safety
X_train_dt, X_test_dt = X_train_raw.copy(), X_test_raw.copy()


# In[74]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Handle missing values using mean
imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

print(y_pred)


# In[75]:


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[76]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Create copies specific to KNN to preserve pure data structures
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert back to dataframes/arrays for structural safety
X_train_dt, X_test_dt = X_train_raw.copy(), X_test_raw.copy()


# In[77]:


# Step 17: List current columns
print(tree.columns.tolist())


# In[78]:


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_preds)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Decision Tree Confusion Matrix')
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')

# KNN Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_preds)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('KNN Confusion Matrix')
ax[1].set_xlabel('Predicted Label')
ax[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()


# In[79]:


# Step 27: Check if variables like distance or bnp have anomalies (Replacing Billing Amount logic)
print("Number of negative distances:", (tree['distance_to_hospital_km'] < 0).sum())


# In[80]:


# Step 28 & 29: Filter out any anomalous rows if they exist, create clean dataset
# (Adapting 'Billing Amount' filtering to 'distance_to_hospital_km')
anomalous_records = tree[tree['distance_to_hospital_km'] < 0].copy()
tree_clean = tree[tree['distance_to_hospital_km'] >= 0].copy()


# In[81]:


print("Original dataset size:", tree.shape[0])
print("Clean dataset size:", tree_clean.shape[0])
print(tree_clean.head())


# In[82]:


from sklearn.preprocessing import LabelEncoder

# List of categorical columns
cat_cols = [ 'age', 'bmi', 'bnp']

# Apply Label Encoding to each column
for col in cat_cols:
    le = LabelEncoder()
    tree[col + '_encoded'] = le.fit_transform(tree[col])
    print(f"{col} mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


# In[83]:


tree.head()


# In[86]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(tree_clean['age'], bins=20, kde=True)
plt.title('age Distribution')
plt.xlabel('age')
plt.ylabel('Count')
plt.show()

# Optional: descriptive stats
print(tree_clean['age'].describe())


# In[87]:


tree_clean['age_group'] = pd.cut(
    tree_clean['age'],
    bins=[0, 18, 35, 52, 68, 100],
    labels=False
)

# Check distribution in each bucket
print(tree_clean['age_group'].value_counts().sort_index())


# In[88]:


tree_clean.head()


# In[56]:


numeric_cols = tree.select_dtypes(include=[np.number]).columns.tolist()
if 'readmitted_30d' in numeric_cols:
    numeric_cols.remove('readmitted_30d')
categorical_cols = tree.select_dtypes(exclude=[np.number]).columns.tolist()


# In[57]:


for col in numeric_cols:
    tree[col] = tree[col].fillna(tree[col].median())
for col in categorical_cols:
    tree[col] = tree[col].fillna(tree[col].mode()[0])


# In[58]:


for col in numeric_cols:
    Q1 = tree[col].quantile(0.25)
    Q3 = tree[col].quantile(0.75)
    IQR = Q3 - Q1
    tree[col] = np.where(tree[col] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR, tree[col])
    tree[col] = np.where(tree[col] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, tree[col])


# In[59]:


tree_encoded = pd.get_dummies(tree, columns=categorical_cols, drop_first=True)


# In[60]:


# Separate features and target
X = tree_encoded.drop(columns=['readmitted_30d'])
y = tree_encoded['readmitted_30d']

# Split dataset into 80% Train and 20% Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features ONLY for KNN (keeps DT completely unswayed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# In[61]:


# --- Model A: Decision Tree Classifier ---
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_raw, y_train)
dt_preds = dt_model.predict(X_test_raw)

# --- Model B: K-Nearest Neighbors Classifier ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)


# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report # Ensure this is imported

# --- Decision Tree Report ---
print("=========================================")
print("       DECISION TREE CLASSIFICATION REPORT")
print("=========================================")
print(classification_report(y_test, dt_preds, digits=4))

# --- KNN Report ---
print("\n=========================================")
print("          KNN CLASSIFICATION REPORT")
print("=========================================")
print(classification_report(y_test, knn_preds, digits=4))

# --- Feature Importance ---
# Since you used a Decision Tree (dt_model), we use that to plot feature importance.
# We extract the feature names from X_train_raw (or tree_encoded.drop columns).

features = X_train_raw.columns
importances = pd.Series(dt_model.feature_importances_, index=features)

importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance (Decision Tree)')
plt.show()


# In[91]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Prepare Features and Target ---
# 'tree' is your DataFrame. We switch the target to 'readmitted_30d' to fix the error.
features = [col for col in tree.columns if col != 'readmitted_30d']
X = tree[features]
y = tree['readmitted_30d']

# --- 2. Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Initialize Decision Tree Model ---
# Kept as 'tree_model' so it doesn't overwrite your 'tree' dataset variable
tree_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# --- 4. Train Model ---
tree_model.fit(X_train, y_train)

# --- 5. Make Predictions and Evaluate ---
tree_pred = tree_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, tree_pred, digits=4))

# --- 6. Plot Tree (First 3 levels for clean readability) ---
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=['Not Readmitted', 'Readmitted'], # Correct target categories
    filled=True,
    max_depth=3
)
plt.show()


# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Calculate feature importances from your trained dt_model
importances = dt_model.feature_importances_
feature_names = X.columns
feat_imp_dict = dict(zip(feature_names, importances))

# 2. Categorize variables into the main fishbone branches
categories = {
    "Patient Demographics": ["age", "gender_Male", "bmi"],
    "Vital Signs": ["systolic_bp", "heart_rate"],
    "Lab Tests": ["bnp", "sodium", "creatinine"],
    "Prescribed Medications": ["ace_inhibitor", "beta_blocker", "diuretic"],
    "Social & Adherence Factors": ["adherence_score", "income_level_Low", "income_level_Medium", "distance_to_hospital_km"]
}

# 3. Initialize the plot canvas
plt.figure(figsize=(16, 10))
ax = plt.gca()
ax.set_facecolor('#f8f9fa')  # Clean soft background

# Draw the main backbone (central horizontal line)
plt.annotate('', xy=(10, 5), xytext=(1, 5),
             arrowprops=dict(arrowstyle="->", color='#0f4c5c', lw=5))

# Draw the Fish Head (The Problem Statement)
head_poly = plt.Polygon([[10, 5], [11.5, 6.2], [12, 5], [11.5, 3.8]], color='#2a9d8f')
ax.add_patch(head_poly)
plt.text(11.1, 5, "30-Day\nReadmission\n(Target)",
         ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# Draw the Fish Tail
tail_poly = plt.Polygon([[1, 5], [0.2, 6.2], [0.5, 5], [0.2, 3.8]], color='#e76f51')
ax.add_patch(tail_poly)

# 4. Plot major branches and map features dynamically
up_x_positions = [3, 5.5, 8]      # Horizontal positions for upper bones
down_x_positions = [4.2, 7]       # Horizontal positions for lower bones

cat_keys = list(categories.keys())

# Plot Upper Bones
for i, x_pos in enumerate(up_x_positions):
    if i < len(cat_keys):
        cat_name = cat_keys[i]
        # Draw main diagonal branch line
        plt.plot([x_pos, x_pos + 1.2], [5, 7.5], color='#457b9d', lw=3)
        # Add category title box
        plt.text(x_pos + 1.2, 7.7, cat_name, ha='center', va='bottom', fontsize=11,
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc='#e1edd0', ec='#457b9d', lw=1.5))
       
        # Add sub-features along the bone based on their importance
        y_offset = 5.6
        for feat in categories[cat_name]:
            matching_feats = [f for f in feat_imp_dict if f.startswith(feat)]
            imp_val = sum([feat_imp_dict[mf] for mf in matching_feats]) if matching_feats else 0.0
           
            text_label = f"{feat.replace('_', ' ')} ({imp_val:.2%})"
            current_x = x_pos + ((y_offset - 5) * 1.2 / 2.5)
            plt.annotate('', xy=(current_x, y_offset), xytext=(current_x - 1, y_offset),
                         arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5))
            plt.text(current_x - 1.05, y_offset + 0.05, text_label, ha='left', va='bottom', fontsize=9.5)
            y_offset += 0.6

# Plot Lower Bones
for i, x_pos in enumerate(down_x_positions):
    idx = i + 3  
    if idx < len(cat_keys):
        cat_name = cat_keys[idx]
        # Draw main diagonal branch line pointing downwards
        plt.plot([x_pos, x_pos + 1.2], [5, 2.5], color='#457b9d', lw=3)
        # Add category title box
        plt.text(x_pos + 1.2, 2.3, cat_name, ha='center', va='top', fontsize=11,
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc='#fceade', ec='#e76f51', lw=1.5))
       
        # Add sub-features along the bone
        y_offset = 4.4
        for feat in categories[cat_name]:
            matching_feats = [f for f in feat_imp_dict if f.startswith(feat)]
            imp_val = sum([feat_imp_dict[mf] for mf in matching_feats]) if matching_feats else 0.0
           
            text_label = f"{feat.replace('_', ' ')} ({imp_val:.2%})"
            current_x = x_pos + ((5 - y_offset) * 1.2 / 2.5)
            plt.annotate('', xy=(current_x, y_offset), xytext=(current_x - 1, y_offset),
                         arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5))
            plt.text(current_x - 1.05, y_offset + 0.05, text_label, ha='left', va='bottom', fontsize=9.5)
            y_offset -= 0.6

# 5. Calculate metrics for the summary box
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Add Model Performance Summary box
performance_text = (
    "📊 Decision Tree Performance:\n"
    f"• Accuracy: {accuracy:.2%}\n"
    f"• Precision: {precision:.2%}\n"
    f"• Recall: {recall:.2%}\n"
    f"• F1-Score: {f1:.2%}"
)
plt.text(12, 1.5, performance_text, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.6", fc='#ffffff', ec='#2a9d8f', lw=2))

# Final Layout Enhancements
plt.title("Ishikawa (Fishbone) Diagram of Heart Failure Readmission Causes\nBased on Decision Tree Feature Importances",
          fontsize=14, fontweight='bold', pad=20, color='#1d3557', ha='center')

plt.xlim(0, 13)
plt.ylim(1, 9)
plt.axis('off')  # Hide grid axes
plt.tight_layout()

# Save the final image to your current workspace directory
plt.savefig('fishbone_diagram_en.png', dpi=300)
plt.show()


# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Prepare Features and Target ---
# 'tree' is your DataFrame. We switch the target to 'readmitted_30d' to fix the error.
features = [col for col in tree.columns if col != 'readmitted_30d']
X = tree[features]
y = tree['readmitted_30d']

# --- 2. Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Initialize Decision Tree Model ---
# Kept as 'tree_model' so it doesn't overwrite your 'tree' dataset variable
tree_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# --- 4. Train Model ---
tree_model.fit(X_train, y_train)

# --- 5. Make Predictions and Evaluate ---
tree_pred = tree_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, tree_pred, digits=4))

# --- 6. Plot Tree (First 3 levels for clean readability) ---
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=['Not Readmitted', 'Readmitted'], # Correct target categories
    filled=True,
    max_depth=3
)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Clean Data & Convert Text Categories to Numbers ---
# Ensure unique identifier column is dropped
if 'patient_id' in tree.columns:
    tree = tree.drop(columns=['patient_id'])

# Handle missing numeric values automatically so it runs clean
tree = tree.fillna(tree.median(numeric_only=True))

# Convert text variables (gender, income_level) to numerical 0s and 1s
tree_encoded = pd.get_dummies(tree, drop_first=True)

# --- 2. Prepare Features and Target ---
# Using 'readmitted_30d' from your updated columns
features = [col for col in tree_encoded.columns if col != 'readmitted_30d']
X = tree_encoded[features].astype(float)
y = tree_encoded['readmitted_30d']

# --- 3. Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. Initialize and Train Decision Tree Model ---
tree_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
tree_model.fit(X_train, y_train)

# --- 5. Make Predictions and Evaluate ---
tree_pred = tree_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, tree_pred, digits=4))

# --- 6. Plot Tree Diagram (First 3 levels for clean readability) ---
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=['Not Readmitted', 'Readmitted'],
    filled=True,
    max_depth=3
)
plt.show()


# In[ ]:




