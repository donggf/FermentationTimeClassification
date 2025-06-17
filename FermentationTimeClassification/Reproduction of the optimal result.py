import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, f_classif, SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold,train_test_split,GridSearchCV, cross_validate,KFold

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define residual block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),  # Fully connected layer
            nn.ReLU(),            # Activation function
            nn.Linear(dim, dim),  # Fully connected layer
        )
    def forward(self, x):
        return x + self.layers(x)  # Residual connection

# Define simplified denoising model (with reduced number of residual blocks)
class DenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingModel, self).__init__()
        self.initial = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Initial fully connected layer
        self.res_block = ResidualBlock(hidden_dim)  # Only one residual block
        self.output = nn.Linear(hidden_dim, input_dim)  # Output layer
        self.time_embedding = nn.Linear(1, hidden_dim)  # Time embedding layer

    def forward(self, x, t):
        t = t.view(-1, 1).float().to(device)  # Adjust time dimension
        t_embed = self.time_embedding(t)      # Generate time embedding
        x_input = torch.cat([x, t_embed], dim=1)  # Concatenate input and time embedding
        h = F.relu(self.initial(x_input))     # Initial layer output
        h = self.res_block(h)                 # Pass through residual block
        return self.output(h)                 # Return denoised result

# Forward diffusion process (add noise)
def forward_diffusion(x0, t, noise_schedule):
    batch_size = x0.shape[0]
    sqrt_alpha_t = torch.sqrt(noise_schedule['alpha_bar'][t]).view(batch_size, 1).to(device)
    one_minus_alpha_t = torch.sqrt(1 - noise_schedule['alpha_bar'][t]).view(batch_size, 1).to(device)
    noise = torch.randn_like(x0).to(device)  # Generate random noise
    xt = sqrt_alpha_t * x0 + one_minus_alpha_t * noise  # Noisy data
    return xt, noise

# Create cosine noise schedule
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps).to(device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)  # Limit beta value range

def create_noise_schedule(timesteps):
    betas = cosine_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar}  # Return noise schedule parameters

# Train diffusion model (with learning rate scheduling and L2 regularization)
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_diffusion(model, data, timesteps=500, num_epoch=100, lr=0.001):
    data = data.astype(np.float32)  # Convert to float
    if np.isnan(data).any():
        data = np.nan_to_num(data)  # Handle missing values
    data_tensor = torch.from_numpy(data).float().to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Add L2 regularization
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)  # Learning rate scheduler
    noise_schedule = create_noise_schedule(timesteps)

    for epoch in range(num_epoch):
        total_loss = 0
        for i in range(0, len(data_tensor), 32):  # Batch size of 32
            batch_data = data_tensor[i:i+32]
            batch_size = batch_data.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device)  # Random timestep
            xt, noise = forward_diffusion(batch_data, t, noise_schedule)  # Forward diffusion
            optimizer.zero_grad()
            predicted_noise = model(xt, t)  # Predict noise
            loss = F.mse_loss(predicted_noise, noise)  # Calculate MSE loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()  # Update learning rate
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / (len(data_tensor) / 32)
            print(f'Epoch {epoch+1}, average loss: {avg_loss}')
    return model, avg_loss

# Generate synthetic samples
# Modified generation function to avoid randomness, fix generator device consistency issues, and replace unsupported generator function calls
def generate_synthetic_samples(model, num_samples, input_dim, timesteps=500):
    model = model.to(device)
    model.eval()
    noise_schedule = create_noise_schedule(timesteps)

    # Use generator consistent with current device (avoid device mismatch)
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(num_samples, input_dim, generator=generator, device=device) * 1.5

    with torch.inference_mode():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device)
            predicted_noise = model(x, t_tensor)
            alpha_t = noise_schedule['alphas'][t]
            alpha_bar_t = noise_schedule['alpha_bar'][t]
            beta_t = noise_schedule['betas'][t]

            if t > 0:
                # Also create generator for current device
                gen = torch.Generator(device=device).manual_seed(42 + t)
                # Use torch.randn to generate random tensor with same shape as x, replacing torch.randn_like
                z = torch.randn(x.size(), generator=gen, device=x.device)
            else:
                z = 0

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * z

        synthetic_samples_continuous = torch.sigmoid(x)
        synthetic_samples = torch.bernoulli(synthetic_samples_continuous).cpu().numpy()
    return synthetic_samples



# ------------------ Main process begins ------------------
# Original data loading (kept as is)

# Load training data
vec1 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集1.csv', low_memory=False)
vec2 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集2.csv', low_memory=False)
vec3 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集3.csv', low_memory=False)
vec = np.concatenate([vec1.values, vec2.values, vec3.values], axis=1)
vec = vec[1:, ]  # Remove first row (assuming it's header)
vec = pd.DataFrame(vec)
vec = vec.apply(pd.to_numeric, errors='coerce')  # Convert to numeric type
vec = vec.dropna()  # Remove missing values

vec_label = np.loadtxt("data/Trainingset/train_label.csv", delimiter=",")
print("Label array:", vec_label)
print("Label shape:", vec_label.shape)
num_samples = len(vec)
random_indices = np.random.permutation(num_samples)  # Randomly shuffle indices
shuffled_vec = vec.iloc[random_indices]
shuffled_vec_label = vec_label[random_indices]

# Feature selection pipeline
feature_selection_pipeline = Pipeline([
    ('vt', VarianceThreshold()),
    ('skb2', SelectPercentile(chi2, percentile=4.32)),
    ('skb1', SelectPercentile(f_classif, percentile=50)),
    ('sfm', SelectFromModel(SVC(kernel='linear', C=0.15, random_state=42), threshold=0.0565))
])
vec_selected = feature_selection_pipeline.fit_transform(shuffled_vec.values, shuffled_vec_label)

print("Selected features shape:", vec_selected.shape)
n_features = vec_selected.shape[1]
# Train initial SVM classifier

svm_classifier = Pipeline([
    ('classifier', SVC(kernel='rbf', C=21.224536734693874, gamma=0.05022481203007519, probability=True))
])
svm_classifier.fit(vec_selected, shuffled_vec_label)
scores = cross_val_score(svm_classifier, vec_selected, shuffled_vec_label, cv=10, scoring='roc_auc', n_jobs=-1)
scores1 = cross_val_score(svm_classifier, vec_selected, shuffled_vec_label, cv=10, scoring='accuracy', n_jobs=-1)
print("AUC", scores.mean())
print("ACC", scores1.mean())


# Load prediction set data and predict
Bvec1 = np.loadtxt(open("data/Unlabeleddata/01data/预测集数据集1.csv", "rb"), delimiter=",")
Bvec2 = np.loadtxt(open("data/Unlabeleddata/01data/预测集数据集2.csv", "rb"), delimiter=",")
Bvec3 = np.loadtxt(open("data/Unlabeleddata/01data/预测集数据集3.csv", "rb"), delimiter=",")
Bvec = np.concatenate([Bvec1, Bvec2, Bvec3], axis=1)
Bvec_selected = feature_selection_pipeline.transform(Bvec)
predicted_probs = svm_classifier.predict_proba(Bvec_selected)
predicted_labels = svm_classifier.predict(Bvec_selected)

# Filter high-confidence samples
high_confidence_indices = np.where(np.max(predicted_probs, axis=1) > 0.8)[0]
high_confidence_features = Bvec_selected[high_confidence_indices]
high_confidence_labels = predicted_labels[high_confidence_indices]

# Augment dataset
augmented_features = np.vstack([vec_selected, high_confidence_features])
augmented_labels = np.concatenate([shuffled_vec_label, high_confidence_labels])
print("Augmented dataset shape:", augmented_features.shape)
print("Augmented labels shape:", augmented_labels.shape)

class0_augmented = augmented_features[augmented_labels == 0]  # Class 0 samples
class1_augmented = augmented_features[augmented_labels == 1]  # Class 1 samples
n_features = vec_selected.shape[1]

# Use optimized parameters directly
DIFFUSION_PARAMS = {
    'class0': {'hidden_dim': 5, 'lr': 0.00093485, 'epochs': 164},
    'class1': {'hidden_dim': 19, 'lr': 0.0037, 'epochs': 80}
}

SVM_PARAMS = {
    'a1': 100,  # Number of synthetic samples for class 0
    'a2': 117,  # Number of synthetic samples for class 1
    'C': 4.35400148,
    'gamma': 0.0192,
    'kernel': 'rbf'
}

# Split augmented data by class
class0_augmented = augmented_features[augmented_labels == 0]
class1_augmented = augmented_features[augmented_labels == 1]
n_features = vec_selected.shape[1]



hidden_dim0 = 5
lr0_fixed = 0.00093485
num_epoch0 = 164
diffusion_model0 = DenoisingModel(n_features, hidden_dim0)
diffusion_model0, _ = train_diffusion(diffusion_model0, class0_augmented, timesteps=300, num_epoch=num_epoch0, lr=lr0_fixed)

hidden_dim1 =  19
lr1_fixed =0.0037
num_epoch1 = 80
diffusion_model1 = DenoisingModel(n_features, hidden_dim1)
diffusion_model1, _ = train_diffusion(diffusion_model1, class1_augmented, timesteps=300, num_epoch=num_epoch1, lr=lr1_fixed)


a1_fixed = 100
a2_fixed = 116
C_fixed =  round(float(4.35400148), 2)
gamma_fixed =  round(float(0.02267708), 2)
kernel_fixed = 'rbf'

# Generate synthetic samples (all using timesteps=300, consistent with diffusion model training)
synthetic_class0 = generate_synthetic_samples(diffusion_model0, a1_fixed, n_features, timesteps=300)
synthetic_class1 = generate_synthetic_samples(diffusion_model1, a2_fixed, n_features, timesteps=300)

# Combine real and synthetic samples
class0_all = np.vstack([class0_augmented, synthetic_class0])
class1_all = np.vstack([class1_augmented, synthetic_class1])
resampled_features = np.vstack([class0_all, class1_all])
resampled_labels = np.array([0] * len(class0_all) + [1] * len(class1_all))

# Train final classifier with fixed SVM parameters
svm_classifier_final = Pipeline([
    ('classifier', SVC(kernel=kernel_fixed, C=C_fixed, gamma=gamma_fixed, probability=True))
])
svm_classifier_final.fit(resampled_features, resampled_labels)

# ---------------------- Validation phase ----------------------
# Load validation set data
Pvec1 = np.loadtxt(open("data/Testset/01data/验证集数据集1.csv", "rb"), delimiter=",")
Pvec2 = np.loadtxt(open("data/Testset/01data/验证集数据集2.csv", "rb"), delimiter=",")
Pvec3 = np.loadtxt(open("data/Testset/01data/验证集数据集3.csv", "rb"), delimiter=",")
Pvec = np.concatenate([Pvec1, Pvec2, Pvec3], axis=1)
selected_features = feature_selection_pipeline.transform(Pvec)
Pvec_label = np.loadtxt("data/Testset/testlabel.csv", delimiter=",")

# Prediction and evaluation
predicted_label_final = svm_classifier_final.predict(selected_features)
final_score = accuracy_score(Pvec_label, predicted_label_final)
print("Final predictions:", predicted_label_final)
print("Final validation set accuracy:", final_score)