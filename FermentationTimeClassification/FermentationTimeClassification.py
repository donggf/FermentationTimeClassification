import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizationalgorithm.DBO import DBO  # Assuming DBO optimization algorithm is defined

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
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return x + self.layers(x)

# Define denoising model
class DenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingModel, self).__init__()
        self.initial = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.res_block = ResidualBlock(hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.time_embedding = nn.Linear(1, hidden_dim)

    def forward(self, x, t):
        t = t.view(-1, 1).float().to(device)
        t_embed = self.time_embedding(t)
        x_input = torch.cat([x, t_embed], dim=1)
        h = F.relu(self.initial(x_input))
        h = self.res_block(h)
        return self.output(h)

# Forward diffusion process
def forward_diffusion(x0, t, noise_schedule):
    batch_size = x0.shape[0]
    sqrt_alpha_t = torch.sqrt(noise_schedule['alpha_bar'][t]).view(batch_size, 1).to(device)
    one_minus_alpha_t = torch.sqrt(1 - noise_schedule['alpha_bar'][t]).view(batch_size, 1).to(device)
    noise = torch.randn_like(x0).to(device)
    xt = sqrt_alpha_t * x0 + one_minus_alpha_t * noise
    return xt, noise

# Sigmoid noise schedule
def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
    steps = torch.linspace(0, 1, timesteps).to(device)
    betas = start + (end - start) * (1 / (1 + torch.exp(-10 * (steps - 0.5))))
    return betas

def create_noise_schedule(timesteps):
    betas = sigmoid_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar}

# Train diffusion model
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_diffusion(model, data, timesteps=300, num_epoch=100, lr=0.001):
    data = data.astype(np.float32)
    if np.isnan(data).any():
        data = np.nan_to_num(data)
    data_tensor = torch.from_numpy(data).float().to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)
    noise_schedule = create_noise_schedule(timesteps)

    for epoch in range(num_epoch):
        total_loss = 0
        for i in range(0, len(data_tensor), 32):
            batch_data = data_tensor[i:i+32]
            batch_size = batch_data.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            xt, noise = forward_diffusion(batch_data, t, noise_schedule)
            optimizer.zero_grad()
            predicted_noise = model(xt, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / (len(data_tensor) / 32)
            #print(f'Epoch {epoch+1}, Average loss: {avg_loss}')
    return model, avg_loss

# Generate synthetic samples
def generate_synthetic_samples(model, num_samples, input_dim, timesteps=300):
    model = model.to(device)
    model.eval()
    noise_schedule = create_noise_schedule(timesteps)
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
                gen = torch.Generator(device=device).manual_seed(42 + t)
                z = torch.randn(x.size(), generator=gen, device=device)
            else:
                z = 0

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * z
            x = torch.clamp(x, -10, 10)

        synthetic_samples_continuous = torch.sigmoid(x)
        synthetic_samples_continuous = torch.clamp(synthetic_samples_continuous, 0, 1)
        synthetic_samples = torch.bernoulli(synthetic_samples_continuous).cpu().numpy()
    return synthetic_samples

# Stage 1: Fitness function for optimizing diffusion model parameters using DBO
def diffusion_fitness_function(params, input_dim, class_data):
    hidden_dim = int(params[0])
    lr = float(params[1])
    num_epoch = int(params[2])
    model = DenoisingModel(input_dim, hidden_dim)
    _, loss = train_diffusion(model, class_data, timesteps=300, num_epoch=num_epoch, lr=lr)
    return loss

# Load training data
vec1 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集1.csv', low_memory=False)
vec2 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集2.csv', low_memory=False)
vec3 = pd.read_csv('data/Trainingset/01data/发酵时间分类数据集3.csv', low_memory=False)
vec = np.concatenate([vec1.values, vec2.values, vec3.values], axis=1)
vec = vec[1:, ]
vec = pd.DataFrame(vec)
vec = vec.apply(pd.to_numeric, errors='coerce')
vec = vec.dropna()

vec_label = np.loadtxt("data/Trainingset/train_label.csv", delimiter=",")
print("Label array:", vec_label)
print("Label shape:", vec_label.shape)
num_samples = len(vec)
random_indices = np.random.permutation(num_samples)
shuffled_vec = vec.iloc[random_indices]
shuffled_vec_label = vec_label[random_indices]

# Feature selection pipeline
feature_selection_pipeline = Pipeline([
    ('vt', VarianceThreshold()),
    ('skb2', SelectPercentile(chi2, percentile=4.32)),
    ('skb1', SelectPercentile(f_classif, percentile=50)),
    ('sfm', SelectFromModel(svm.SVC(kernel='linear', C=0.15, random_state=42), threshold=0.0565))
])
vec_selected = feature_selection_pipeline.fit_transform(shuffled_vec.values, shuffled_vec_label)
print("Selected features shape:", vec_selected.shape)

# Train initial SVM classifier
svm_classifier = Pipeline([
    ('classifier', svm.SVC(kernel='rbf', C=21.224536734693874, gamma=0.05022481203007519, probability=True))
])
svm_classifier.fit(vec_selected, shuffled_vec_label)

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

class0_augmented = augmented_features[augmented_labels == 0]
class1_augmented = augmented_features[augmented_labels == 1]
n_features = vec_selected.shape[1]

# Stage 1: Optimize diffusion model parameters
diffusion_lb0 = [3, 0.00093, 155]  # Lower bounds: hidden_dim, lr, num_epoch
diffusion_ub0 = [7, 0.00094, 170]  # Upper bounds: hidden_dim, lr, num_epoch

diffusion_lb1 = [18, 0.0036, 60]  # Lower bounds: hidden_dim, lr, num_epoch
diffusion_ub1 = [25, 0.004, 90]  # Upper bounds: hidden_dim, lr, num_epoch

# Optimize diffusion model for class 0
diffusion_GbestScore0, diffusion_best_params0, _ = DBO(
    pop=30, dim=3, lb=diffusion_lb0, ub=diffusion_ub0, MaxIter=10,
    fun=lambda x: diffusion_fitness_function(x, n_features, class0_augmented),
    b=0.4, k=0.2, rate=0.5)
print("Class 0 diffusion model best loss:", diffusion_GbestScore0)
print("Class 0 diffusion model best params (hidden_dim, lr, num_epoch):", diffusion_best_params0)

if diffusion_best_params0.ndim > 1:
    diffusion_best_params0 = diffusion_best_params0.flatten()
hidden_dim0 = int(diffusion_best_params0[0])
lr0 = float(diffusion_best_params0[1])
num_epoch0 = int(diffusion_best_params0[2])

# Train best diffusion model for class 0
diffusion_model0 = DenoisingModel(n_features, hidden_dim0)
diffusion_model0, _ = train_diffusion(diffusion_model0, class0_augmented, timesteps=300, num_epoch=num_epoch0, lr=lr0)
#torch.save(diffusion_model0.state_dict(), 'diffusion_model0_best.pth')

# Optimize diffusion model for class 1
diffusion_GbestScore1, diffusion_best_params1, _ = DBO(
    pop=30, dim=3, lb=diffusion_lb1, ub=diffusion_ub1, MaxIter=10,
    fun=lambda x: diffusion_fitness_function(x, n_features, class1_augmented),
    b=0.4, k=0.2, rate=0.5)
print("Class 1 diffusion model best loss:", diffusion_GbestScore1)
print("Class 1 diffusion model best params (hidden_dim, lr, num_epoch):", diffusion_best_params1)

if diffusion_best_params1.ndim > 1:
    diffusion_best_params1 = diffusion_best_params1.flatten()
hidden_dim1 = int(diffusion_best_params1[0])
lr1 = float(diffusion_best_params1[1])
num_epoch1 = int(diffusion_best_params1[2])

# Train best diffusion model for class 1
diffusion_model1 = DenoisingModel(n_features, hidden_dim1)
diffusion_model1, _ = train_diffusion(diffusion_model1, class1_augmented, timesteps=300, num_epoch=num_epoch1, lr=lr1)
#torch.save(diffusion_model1.state_dict(), 'diffusion_model1_best.pth')

# Load validation set data
Pvec1 = np.loadtxt(open("data/Testset/01data/验证集数据集1.csv", "rb"), delimiter=",")
Pvec2 = np.loadtxt(open("data/Testset/01data/验证集数据集2.csv", "rb"), delimiter=",")
Pvec3 = np.loadtxt(open("data/Testset/01data/验证集数据集3.csv", "rb"), delimiter=",")
Pvec = np.concatenate([Pvec1, Pvec2, Pvec3], axis=1)
selected_features = feature_selection_pipeline.transform(Pvec)
Pvec_label = np.loadtxt("data/Testset/testlabel.csv", delimiter=",")

# Define kernel function mapping
kernel_mapping = {1: 'linear', 2: 'rbf', 3: 'sigmoid', 4: 'poly'}

# Stage 2: Fitness function for optimizing SVM parameters using optimized diffusion models (based on training set CV)
def svm_fitness_function(params):
    a1 = int(params[0])  # Number of synthetic samples for class 0
    a2 = int(params[1])  # Number of synthetic samples for class 1
    C = round(float(params[2]), 2)  # SVM regularization parameter
    gamma = round(float(params[3]), 2)  # SVM kernel parameter
    kernel_index = int(params[4])  # Kernel function index
    kernel = kernel_mapping[kernel_index]

    # Generate synthetic samples
    if a1 > 0:
        synthetic_class0 = generate_synthetic_samples(diffusion_model0, a1, n_features, timesteps=300)
    else:
        synthetic_class0 = np.empty0, n_features
    if a2 > 0:
        synthetic_class1 = generate_synthetic_samples(diffusion_model1, a2, n_features, timesteps=300)
    else:
        synthetic_class1 = np.empty0, n_features

    # Initialize cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    # Train and evaluate for each fold
    for train_index, test_index in kf.split(augmented_features):
        # Split current fold's train and test sets
        X_train = augmented_features[train_index]
        y_train = augmented_labels[train_index]
        X_test = augmented_features[test_index]
        y_test = augmented_labels[test_index]

        # Split classes based on current fold's training set
        class0_train = X_train[y_train == 0]
        class1_train = X_train[y_train == 1]

        # Combine current fold's training set with synthetic samples
        class0_all = np.vstack([class0_train, synthetic_class0])
        class1_all = np.vstack([class1_train, synthetic_class1])
        resampled_features = np.vstack([class0_all, class1_all])
        resampled_labels = np.array([0] * len(class0_all) + [1] * len(class1_all))

        # Train SVM classifier
        svm_classifier = Pipeline([
            ('classifier', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
        ])
        svm_classifier.fit(resampled_features, resampled_labels)

        # Evaluate on current fold's test set
        y_pred = svm_classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    return -np.mean(scores)  # Return negative average accuracy for minimization


# Stage 2: Optimize SVM parameters
svm_lb = [95, 95, 4.0, 0.025, 1]  # Lower bounds: a1, a2, C, gamma, kernel
svm_ub = [101, 120, 5.0, 0.028, 4]  # Upper bounds: a1, a2, C, gamma, kernel (fixed to rbf)
svm_GbestScore, svm_best_params, svm_Curve = DBO(
    pop=30, dim=5, lb=svm_lb, ub=svm_ub, MaxIter=100,
    fun=svm_fitness_function, b=0.4, k=0.2, rate=0.5)
print("SVM best CV accuracy:", -svm_GbestScore)
print("SVM best params (a1, a2, C, gamma, kernel):", svm_best_params)


if svm_best_params.ndim > 1:
    svm_best_params = svm_best_params.flatten()

# Train final SVM classifier with optimized parameters
a1 = int(svm_best_params[0])
a2 = int(svm_best_params[1])
C = round(float(svm_best_params[2]), 2)
gamma = round(float(svm_best_params[3]), 2)
kernel_index = int(svm_best_params[4])
kernel = kernel_mapping[kernel_index]

# Generate final synthetic samples
synthetic_class0 = generate_synthetic_samples(diffusion_model0, a1, n_features, timesteps=300)
synthetic_class1 = generate_synthetic_samples(diffusion_model1, a2, n_features, timesteps=300)
class0_all = np.vstack([class0_augmented, synthetic_class0])
class1_all = np.vstack([class1_augmented, synthetic_class1])
resampled_features = np.vstack([class0_all, class1_all])
resampled_labels = np.array([0] * len(class0_all) + [1] * len(class1_all))

# Train final SVM classifier
svm_classifier_final = Pipeline([
    ('classifier', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
])
svm_classifier_final.fit(resampled_features, resampled_labels)

# Evaluate on validation set
predicted_label_final = svm_classifier_final.predict(selected_features)
final_score = accuracy_score(Pvec_label, predicted_label_final)
print("Final validation set accuracy:", final_score)