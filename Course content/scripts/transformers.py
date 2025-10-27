import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments

# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=16):
        self.X = torch.randn(n_samples, seq_len, dim)
        # Example: output = mean of inputs
        self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# ----- 2. Small transformer model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=16, model_dim=64, num_heads=4, num_layers=2, output_dim=16):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
    #
    def forward(self, input_vectors, labels=None):
        x = self.input_proj(input_vectors)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # simple mean pooling
        preds = self.output_proj(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----- 3. Trainer-compatible wrappers -----
def collate_fn(batch):
    inputs = torch.stack([item["input_vectors"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_vectors": inputs, "labels": labels}

train_dataset = VectorDataset()
val_dataset = VectorDataset(200)

model = TransformerRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    eval_strategy="epoch", ### or evaluation_strategy # this depends on transformer version
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ----- 4. Train -----
trainer.train()

# ----- 5. Predict -----
test_input = torch.randn(1, 10, 16)
with torch.no_grad():
    output = model(test_input)["logits"]

print("Predicted output vector:", output)




############################### another example ####


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=16):
        # self.X = torch.randn(n_samples, seq_len, dim)
        self.X = (torch.randn(n_samples, seq_len, dim) > 0.5) + 0.0
        # Example: output = mean of inputs
        self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# ----- 2. Small transformer model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=16, model_dim=64, num_heads=4, num_layers=2, output_dim=16):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
    #
    def forward(self, input_vectors, labels=None):
        x = self.input_proj(input_vectors)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # simple mean pooling
        preds = self.output_proj(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----- 3. Trainer-compatible wrappers -----
def collate_fn(batch):
    inputs = torch.stack([item["input_vectors"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_vectors": inputs, "labels": labels}

train_dataset = VectorDataset()
val_dataset = VectorDataset(200)

model = TransformerRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    eval_strategy="epoch", ### or evaluation_strategy # this depends on transformer version
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ----- 4. Train -----
trainer.train()



### 4b ---- plotting -----

import matplotlib.pyplot as plt

# Access log history
logs = trainer.state.log_history

# Extract loss per epoch
train_loss = [entry["loss"] for entry in logs if "loss" in entry]
eval_loss = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
epochs = range(1, len(eval_loss) + 1)

# Plot
plt.figure(figsize=(6,4))
plt.plot(epochs, eval_loss, label="Eval loss", marker='o')
plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training & Evaluation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# ----- 5. Test on new case (reals) -----
test_input = torch.randn(1, 10, 16)
with torch.no_grad():
    output = model(test_input)["logits"]

print("Predicted output vector:", output)
test_input.mean(dim=1)

# ----- 5. Test on trained domain (integers) -----
test_input = (torch.randn(1, 10, 16) > 0.5) + 0.0 ### just +0 doesn't work, check
with torch.no_grad():
    output = model(test_input)["logits"]

print("Predicted output vector:", output)
test_input.mean(dim=1)

#>>> print("Predicted output vector:", output)
#Predicted output vector: 
#tensor([[0.3030, 0.3012, 0.5044, 0.4996, 0.4028, 0.1975, 0.4019, 0.3998, 0.1004,
#         0.3003, 0.1008, 0.0991, 0.2001, 0.3034, 0.4035, 0.4023]])
#>>> test_input.mean(dim=1)
#tensor([[0.3000, 0.3000, 0.5000, 0.5000, 0.4000, 0.2000, 0.4000, 0.4000, 0.1000,
#         0.3000, 0.1000, 0.1000, 0.2000, 0.3000, 0.4000, 0.4000]])


##########  a more interesting problem: circularity (in 2D)


def circularity_score(points: torch.Tensor, eps=1e-8) -> float: ## quite suboptimal!!
    """
    Compute a circularity score ∈ [0,1] for a 2D point cloud.
    #
    Args:
        points (torch.Tensor): shape (N, 2)
        eps (float): numerical stability constant
    #
    Returns:
        float: circularity score (1=circle-like, 0=line-like)
    """
    assert points.ndim == 2 and points.shape[1] == 2, "Expected shape (N,2)"
    #
    # 1. Center points
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    #
    # 2. Compute radius of each point
    radii = torch.norm(centered, dim=1)
    #
    # 3. Measure how uniform radii are
    mean_r = radii.mean()
    std_r = radii.std()
    #
    # 4. Relative std (coefficient of variation)
    eps = 0.01*mean_r
    rel_std = std_r / (mean_r + eps)
    #
    # 5. Convert to score in [0, 1]
    #   - Smaller rel_std → more circular
    #   - Larger rel_std → more line-like
    score = torch.exp(-5 * rel_std)  # tune the 5 to adjust sensitivity
    return float(score)


# Perfect circle
theta = torch.linspace(0, 2 * torch.pi, 100)
circle_points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
print("Circle score:", circularity_score(circle_points))

# Straight line
x = torch.linspace(-1, 1, 100)
line_points = torch.stack([x, torch.zeros_like(x)], dim=1)
print("Line score:", circularity_score(line_points))

# Random blob
random_points = torch.randn(100, 2)
print("Random score:", circularity_score(random_points))




##################### improved version 

import torch

def circularity_score_v2(points: torch.Tensor, eps=1e-8) -> float:
    """
    Compute a circularity score in [0, 1]:
    1 -> perfectly circular
    0 -> perfectly linear
    # 
    Combines:
      - Uniformity of radii (relative std)
      - Isotropy of covariance (eigenvalue ratio)
    """
    assert points.ndim == 2 and points.shape[1] == 2, "Expected shape (N,2)"
    #
  # Center the data
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
  #
    # Radial uniformity
    radii = torch.norm(centered, dim=1)
    mean_r = radii.mean()
    std_r = radii.std()
    rel_std = std_r / (mean_r + eps)  # smaller = more circular
    radial_score = torch.exp(-5 * rel_std)  # smooth 0–1 mapping
  #
  # Covariance isotropy
    cov = torch.cov(centered.T)  # 2x2 covariance matrix
    eigvals = torch.linalg.eigvalsh(cov)  # sorted ascending, real-valued
    lam_min, lam_max = eigvals[0].item(), eigvals[1].item()
    eig_ratio = lam_min / (lam_max + eps)  # 1 = circle, 0 = line
    cov_score = eig_ratio ** 0.5  # optional smoothing
  #
    # Combine both scores
    # Use geometric mean for balanced weighting
    combined_score = torch.sqrt(radial_score * cov_score)
  #
    return float(combined_score)

# Perfect circle
theta = torch.linspace(0, 2 * torch.pi, 100)
circle_points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
print("Circle score:", circularity_score(circle_points))
print("Circle score:", circularity_score_v2(circle_points))

# Straight line
x = torch.linspace(-1, 1, 100)
line_points = torch.stack([x, torch.zeros_like(x)+4], dim=1)
print("Line score:", circularity_score(line_points))
print("Line score:", circularity_score_v2(line_points))

# Random blob
random_points = torch.randn(100, 2)
print("Random score:", circularity_score(random_points))
print("Random score:", circularity_score_v2(random_points))

## use this function
from types import SimpleNamespace

def create_data(n_samples, seq_len):
  res = SimpleNamespace()
  res.X = []
  res.y = []
  #
  for j in range(n_samples):
      if torch.randn(1).item() > 0:
          theta = torch.randn(seq_len, 1)
          res.X.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
          res.y.append(1)
      else:
          tmp = torch.randn(seq_len, 1)
          res.X.append(torch.stack([tmp, tmp + torch.randn(seq_len, 1)], dim=1))
          res.y.append(0)
  #
  # Convert lists to tensors if you like
  res.X = torch.stack(res.X)  # shape: (n_samples, seq_len, 2)
  res.y = torch.tensor(res.y) # shape: (n_samples,)
  return(res)

############### sequence -> boolean output


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        # self.X = torch.randn(n_samples, seq_len, dim)
        X = (torch.randn(n_samples, seq_len, dim) > 0.5) + 0.0
        self.X = X
        y = []
        for i in range(0, n_samples):
             y.append([ X[i][2][0], X[i][2][1] * X[i][3][1], 1-X[i][4][2] ])
             #y.append([ X[i][2][0] ])
        #self.y = torch.tensor(y)
        self.y = torch.tensor(y)
        # Example: output = mean of inputs
        # self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# test = VectorDataset(1000, 10, 3)
# test.X[33].shape
## function above doent work in the test phase


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        self.X = torch.randn(n_samples, seq_len, dim)
        # Example: output = mean of inputs
        self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# ----- 2. Small transformer model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, num_heads=4, num_layers=2, output_dim=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
    #
    def forward(self, input_vectors, labels=None):
        x = self.input_proj(input_vectors)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # simple mean pooling
        preds = self.output_proj(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----- 3. Trainer-compatible wrappers -----
def collate_fn(batch):
    inputs = torch.stack([item["input_vectors"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_vectors": inputs, "labels": labels}

train_dataset = VectorDataset(1000, 10, 3)
val_dataset = VectorDataset(200, 10, 3)

model = TransformerRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1000, # 10,
    eval_strategy="epoch", ### or evaluation_strategy # this depends on transformer version
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ----- 4. Train -----
trainer.train()

# ----- 5. Predict -----
# test_input = torch.randn(1, 10, 16)


test_dataset = VectorDataset(50, 10, 3)

for k in range(0, 10):
  test_input = test_dataset[k]['input_vectors']
  test_input = test_input.unsqueeze(0) # puts an axis in front
  with torch.no_grad():
      output = model(test_input)["logits"]
  #
  print("Predicted: ", output)
  print("Reference: ", test_dataset[k]['labels'])
  print("--------")
  # results are certainly not good yet, but improving

Predicted:  tensor([[ 1.0281, -0.0507,  0.5484]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[ 0.0336, -0.0280,  0.9594]])
Reference:  tensor([0., 1., 1.])
--------
Predicted:  tensor([[-0.1234,  0.0814,  1.0067]])
Reference:  tensor([0., 1., 0.])
--------
Predicted:  tensor([[1.1041, 0.1537, 1.0065]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[1.0987, 0.4113, 0.4853]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[0.3387, 0.0772, 1.0787]])
Reference:  tensor([0., 0., 1.])
--------
Predicted:  tensor([[0.1033, 0.1428, 1.2217]])
Reference:  tensor([0., 1., 1.])

####################### Experiment: another boolean constriction


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        # self.X = torch.randn(n_samples, seq_len, dim)
        X = (torch.randn(n_samples, seq_len, dim) > 0.5) + 0.0
        self.X = X
        y = []
        for i in range(0, n_samples):
             y.append([ 1-X[i][seq_len-1][0], X[i][seq_len-1][1] * X[i][seq_len-2][1], 1-X[i][seq_len-2][2] ])
             #y.append([ X[i][2][0] ])
        #self.y = torch.tensor(y)
        self.y = torch.tensor(y)
        # Example: output = mean of inputs
        # self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# test = VectorDataset(1000, 10, 3)
# test.X[33].shape
## function above doent work in the test phase


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        self.X = torch.randn(n_samples, seq_len, dim)
        # Example: output = mean of inputs
        self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# ----- 2. Small transformer model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, num_heads=4, num_layers=2, output_dim=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
    #
    def forward(self, input_vectors, labels=None):
        x = self.input_proj(input_vectors)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # simple mean pooling
        preds = self.output_proj(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----- 3. Trainer-compatible wrappers -----
def collate_fn(batch):
    inputs = torch.stack([item["input_vectors"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_vectors": inputs, "labels": labels}

train_dataset = VectorDataset(1000, 10, 3)
val_dataset = VectorDataset(200, 10, 3)

model = TransformerRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1000, # 10,
    eval_strategy="epoch", ### or evaluation_strategy # this depends on transformer version
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ----- 4. Train -----
trainer.train()

# ----- 5. Predict -----
# test_input = torch.randn(1, 10, 16)


test_dataset = VectorDataset(50, 10, 3)

for k in range(0, 10):
  test_input = test_dataset[k]['input_vectors']
  test_input = test_input.unsqueeze(0) # puts an axis in front
  with torch.no_grad():
      output = model(test_input)["logits"]
  #
  print("Predicted: ", output)
  print("Reference: ", test_dataset[k]['labels'])
  print("--------")
  # results are certainly not good yet, but improving

Predicted:  tensor([[0.8435, 0.0307, 1.0800]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[0.6193, 0.0174, 1.1564]])
Reference:  tensor([0., 0., 0.])
--------
Predicted:  tensor([[1.0199, 0.0056, 0.7273]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[0.8696, 0.1572, 1.2169]])
Reference:  tensor([0., 0., 1.])
--------
Predicted:  tensor([[1.2198, 0.2389, 0.9272]])
Reference:  tensor([1., 0., 1.])
--------
Predicted:  tensor([[ 1.0629, -0.0662,  0.6617]])
Reference:  tensor([0., 0., 0.])
--------
Predicted:  tensor([[0.3284, 0.0900, 0.6384]])
Reference:  tensor([1., 0., 0.])
--------
Predicted:  tensor([[1.0429, 0.4620, 1.0324]])
Reference:  tensor([1., 0., 1.])

# on validation set:
for k in range(0, 10):
  test_input = val_dataset[k]['input_vectors']
  test_input = test_input.unsqueeze(0) # puts an axis in front
  with torch.no_grad():
      output = model(test_input)["logits"]
  #
  print("Predicted: ", output)
  print("Reference: ", val_dataset[k]['labels'])
  print("--------")
  # results are certainly not good yet


### this shows OK tendencies but is certainly not perfect
### what about more layers? (not really convincing)



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        # self.X = torch.randn(n_samples, seq_len, dim)
        X = (torch.randn(n_samples, seq_len, dim) > 0.5) + 0.0
        self.X = X
        y = []
        for i in range(0, n_samples):
             y.append([ 1-X[i][seq_len-1][0], X[i][seq_len-1][1] * X[i][seq_len-2][1], 1-X[i][seq_len-2][2] ])
             #y.append([ X[i][2][0] ])
        #self.y = torch.tensor(y)
        self.y = torch.tensor(y)
        # Example: output = mean of inputs
        # self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# test = VectorDataset(1000, 10, 3)
# test.X[33].shape
## function above doent work in the test phase


# ----- 1. Dummy dataset -----
class VectorDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=10, dim=3):
        self.X = torch.randn(n_samples, seq_len, dim)
        # Example: output = mean of inputs
        self.y = self.X.mean(dim=1)
    #
    def __len__(self):
        return len(self.X)
    #
    def __getitem__(self, idx):
        return {"input_vectors": self.X[idx], "labels": self.y[idx]}

# ----- 2. Small transformer model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, num_heads=4, num_layers=4, output_dim=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
    #
    def forward(self, input_vectors, labels=None):
        x = self.input_proj(input_vectors)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # simple mean pooling
        #pooled = encoded # cannot be done in this way
        preds = self.output_proj(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----- 3. Trainer-compatible wrappers -----
def collate_fn(batch):
    inputs = torch.stack([item["input_vectors"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_vectors": inputs, "labels": labels}

train_dataset = VectorDataset(1000, 10, 3)
val_dataset = VectorDataset(200, 10, 3)

model = TransformerRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1000, # 10,
    eval_strategy="epoch", ### or evaluation_strategy # this depends on transformer version
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ----- 4. Train -----
trainer.train()

# ----- 5. Predict -----
# test_input = torch.randn(1, 10, 16)


test_dataset = VectorDataset(50, 10, 3)

for k in range(0, 10):
  test_input = test_dataset[k]['input_vectors']
  test_input = test_input.unsqueeze(0) # puts an axis in front
  with torch.no_grad():
      output = model(test_input)["logits"]
  #
  print("Predicted: ", output)
  print("Reference: ", test_dataset[k]['labels'])
  print("--------")

# on training set
for k in range(0, 10):
  test_input = train_dataset[k]['input_vectors']
  test_input = test_input.unsqueeze(0) # puts an axis in front
  with torch.no_grad():
      output = model(test_input)["logits"]
  #
  print("Predicted: ", output)
  print("Reference: ", train_dataset[k]['labels'])
  print("--------")  # reasonable convincing 



########################## other mapping sequence ---> boolean:

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Dataset definition
# ----------------------------
class BooleanSequenceDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=10):
        self.X, self.y = self._generate_data(n_samples, seq_len)
  #
    def _generate_data(self, n_samples, seq_len):
        # Input: binary sequences of shape [n_samples, seq_len, 2]
        X = torch.randint(0, 2, (n_samples, seq_len, 2)).float()
   #
        # Example Boolean expressions for outputs:
        #   y1 = 1 if any bit in the sequence is 1 (OR over all)
        #   y2 = 1 if the number of 1's in first component > number in second component
        y1 = (X.sum(dim=(1, 2)) > 0).float()  # any 1 in the sequence
        y2 = (X[:, :, 0].sum(dim=1) > X[:, :, 1].sum(dim=1)).float()
        y = torch.stack([y1, y2], dim=1).unsqueeze(-1)  # shape [N, 2, 1]
        return X, y
  #
    def __len__(self):
        return len(self.X)
  #
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# 2. Transformer model
# ----------------------------
class BooleanTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=32, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pool across sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),  # because outputs are in [0,1]
        )
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        encoded = self.encoder(x)  # [batch, seq_len, model_dim]
        pooled = encoded.mean(dim=1)  # average pooling over sequence
        out = self.output_proj(pooled)  # [batch, output_dim]
        return out.unsqueeze(-1)  # [batch, output_dim, 1]

# ----------------------------
# 3. Training setup
# ----------------------------
def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss()  # binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * X.size(0)
            val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# ----------------------------
# 4. Train and test
# ----------------------------
#def main():
if 1:
    seq_len = 10
    dataset = BooleanSequenceDataset(n_samples=2000, seq_len=seq_len)
    X_train, X_val, y_train, y_val = train_test_split(dataset.X, dataset.y, test_size=0.2)
    #
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    #
    model = BooleanTransformer()
    train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3)

if 1:
    # Test
    # model.eval()
    with torch.no_grad():
        #X_test = torch.randint(0, 2, (1, seq_len, 2)).float()
        y_pred = model(X_test)
        print(X_test)
        #print(y_pred)
        print("Sample predictions:\n", y_pred.squeeze(-1).round()) # yes




# if __name__ == "__main__":
# main()

################## experiment 2 with booleans

########################## other mapping sequence ---> boolean:

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Dataset definition
# ----------------------------
class BooleanSequenceDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=10):
        self.X, self.y = self._generate_data(n_samples, seq_len)
  #
    def _generate_data(self, n_samples, seq_len):
        # Input: binary sequences of shape [n_samples, seq_len, 2]
        X = torch.randint(0, 2, (n_samples, seq_len, 2)).float()
        #
        # Example Boolean expressions for outputs:
        y = torch.randint(0, 2, (n_samples, 2, 1)).float() # shape [N, 2, 1]
        # 
        for i in range(0, X.shape[0]):
          y[i, 0, 0] = X[i, -1, 0]*X[i, -1, 1]
          y[i, 1, 0] = X[i, -1, 0] + X[i, -1, 1] - X[i, -1, 0]*X[i, -1, 1]
        # original in terms of for loop
        #for i in range(0, X.shape[0]):
        #  y[i, 0, 0] = X[i].sum(dim=0)[0] > 0
        #  y[i, 1, 0] = X[i].sum(dim=0)[0] > X[i].sum(dim=0)[1] 
        # original:
        #y1 = (X.sum(dim=(1, 2)) > 0).float()
        #y2 = (X[:, :, 0].sum(dim=1) > X[:, :, 1].sum(dim=1)).float()
        #y = torch.stack([y1, y2], dim=1).unsqueeze(-1)  # shape [N, 2, 1]
        return X, y
  #
    def __len__(self):
        return len(self.X)
  #
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# 2. Transformer model
# ----------------------------
class BooleanTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=32, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pool across sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),  # because outputs are in [0,1]
        )
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        encoded = self.encoder(x)  # [batch, seq_len, model_dim]
        pooled = encoded.mean(dim=1)  # average pooling over sequence
        out = self.output_proj(pooled)  # [batch, output_dim]
        return out.unsqueeze(-1)  # [batch, output_dim, 1]

# ----------------------------
# 3. Training setup
# ----------------------------
def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss()  # binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * X.size(0)
            val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# ----------------------------
# 4. Train and test
# ----------------------------
#def main():
if 1:
    seq_len = 10
    dataset = BooleanSequenceDataset(n_samples=2000, seq_len=seq_len)
    X_train, X_val, y_train, y_val = train_test_split(dataset.X, dataset.y, test_size=0.2)
    #
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    #
    model = BooleanTransformer()
    train_model(model, train_loader, val_loader, n_epochs=400, lr=1e-3)

if 1:
    # Test
    # model.eval()
    with torch.no_grad():
        #X_test = torch.randint(0, 2, (1, seq_len, 2)).float()
        y_pred = model(X_test)
        print(X_test)
        #print(y_pred)
        print("Sample predictions:\n", y_pred.squeeze(-1).round()) # yes




############ the above is still not convincing - also the shapes appear somewhat illogical.

Divide the work in modular steps.
1) Create dataset for training, validation and testing.



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ConstructSet:
    def __init__(self, n_samples, seq_len, dim, generator_func):
        """
        Create a data set A with A.X and A.y
        - generator_func: a function f(X_k) -> y_k that defines how y_k is computed
        """
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.dim = dim
        #
        # Create random data (can adjust distribution as needed)
        #self.X = np.random.randn(n_samples, seq_len, dim).astype(float)
        self.X = np.random.randint(0, 2, (n_samples, seq_len, dim)).astype(float)
        ## added louis:
   #     self.X = np.array(self.X, dtype=float)
   #     #
        # Compute y using the provided boolean generator
        self.y = np.array([generator_func(self.X[k]) for k in range(n_samples)], dtype=float)

def example_generator(X_k):
    """
    Example boolean logic:
    Let's say y has shape (dim,), where:
      y[i] = 1 if the mean of feature i across the sequence > 0.5
           = 0 otherwise
    """
    #return (X_k.mean(axis=0) > 0.5).astype(float) # this works nicely
    return X_k[-2].astype(float)


# Example usage
#if __name__ == "__main__":
if 1:
    A = ConstructSet(n_samples=1000, seq_len=10, dim=5, generator_func=example_generator)
    print("A.X shape:", A.X.shape)
    print("A.y shape:", A.y.shape)
    print("First example:")
    print("X[0]:\n", A.X[0])
    print("y[0]:", A.y[0])

 #####



## ============================================================
## 1. Dataset construction
## ============================================================
#class ConstructSet:
#   def __init__(self, n_samples, seq_len, dim, generator_func):
#        self.X = np.random.randn(n_samples, seq_len, dim).astype(np.float)
#        self.y = np.array([generator_func(self.X[k]) for k in range(n_samples)], dtype=np.float)
#
## Example generator (you can replace this)
#def example_generator(X_k):
#    return (X_k.mean(axis=0) > 0).astype(np.float)


# ============================================================
# 2. Transformer Model
# ============================================================

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=None, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)
        #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #
        # Output fully connected layer
        self.output_fc = nn.Linear(model_dim, output_dim or input_dim)
        #
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_fc(x)              # → (batch, seq_len, model_dim)
        x = self.transformer(x)           # → (batch, seq_len, model_dim)
        last_token = x[:, -1, :]          # take the last time step’s embedding → (batch, model_dim)
        out = self.output_fc(last_token)  # → (batch, output_dim)
        return out

# ============================================================
# 3. Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()  # or other loss depending on your task
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    train_losses = []
    val_losses = []
    #
    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        #
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # ---------- Validation ----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        #
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        #
    # Return the loss histories for plotting
    return train_losses, val_losses


# ============================================================
# 4. Example Usage
# ============================================================
#if __name__ == "__main__":
if 1:
    # Create train/val/test sets
    dim, seq_len = 3,4 # 5, 10
    train = ConstructSet(10000, seq_len, dim, example_generator)
    val   = ConstructSet(400,  seq_len, dim, example_generator)
    test  = ConstructSet(400,  seq_len, dim, example_generator)
    #
    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(torch.tensor(train.X, dtype=torch.float32), torch.tensor(train.y,  dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(val.X, dtype=torch.float32), torch.tensor(val.y, dtype=torch.float32)), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(torch.tensor(test.X, dtype=torch.float32), torch.tensor(test.y, dtype=torch.float32)), batch_size=batch_size)
    #
    # Model
    model = TransformerRegressor(input_dim=dim, model_dim=64, num_heads=4, num_layers=2, output_dim=dim)
    #
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=1000, lr=1e-3, device=device)
    #
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_versus_val_loss.png', dpi=300)
    plt.show()
    #


############### another option:



import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================
# 1. Data Construction
# ============================================================
def make_dataset(n_samples=10000, seq_len=10, dim=5):
    # Boolean sequences (0 or 1)
    X = np.random.randint(0, 2, size=(n_samples, seq_len, dim)).astype(np.float32)
    # Boolean OR of the last two timesteps per feature
    y = np.logical_or(X[:, -1, :], X[:, -2, :]).astype(np.float32)
    return X, y

# Split train / validation / test
def build_loaders(batch_size=64):
    X, y = make_dataset(10000, 10, 5)
    n_train, n_val = 8000, 1000
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    #
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                             batch_size=batch_size)
    return train_loader, val_loader, test_loader

# ============================================================
# 2. Transformer Model
# ============================================================
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=None, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #
        # Output fully connected layer (no pooling → use last token)
        self.output_fc = nn.Linear(model_dim, output_dim or input_dim)
        #
    def forward(self, x):
        x = self.input_fc(x)         # (batch, seq_len, model_dim)
        x = self.transformer(x)      # (batch, seq_len, model_dim)
        last_token = x[:, -1, :]     # use last token embedding
        out = self.output_fc(last_token)
        return out

# ============================================================
# 3. Training Function
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    train_losses, val_losses = [], []
    #
    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        #
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        #
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        #
    return train_losses, val_losses

# ============================================================
# 4. Main Script
# ============================================================
#if __name__ == "__main__":
if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = build_loaders(batch_size=64)
    model = TransformerRegressor(input_dim=5, model_dim=64, num_heads=4, num_layers=2, output_dim=5)
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=250, lr=1e-3, device=device)

if 1:
    # Plot loss curves
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('train_versus_val_loss_v2.png', dpi=300)
    plt.show()


if 1:
    # ---- Evaluate on Test Set ----
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
            preds.append(outputs > 0.5)
            trues.append(y_batch.numpy())
    #
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = (preds == trues).mean()
    print(f"\nTest Accuracy: {acc:.4f}")  # 0.7764 after 250 epochs



# test
if 1:
    # Test Evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
            preds.append(outputs)
            trues.append(y_batch.numpy())
    #
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = ((preds > 0.5) == trues).mean()
    print(f"Test Accuracy: {acc:.4f}")





######################### MLP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Define the MLP model
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=5, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    #
    def forward(self, x):
        # x: (batch, seq_len, dim)
        x = x.view(x.size(0), -1)  # flatten sequence (batch, seq_len * dim)
        return self.net(x)

# ============================================================
# 2. Training with Early Stopping
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, patience=10, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    #
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        #
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        #
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        #
        # Early stopping
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                model.load_state_dict(best_model_state)
                break
    #
    return train_losses, val_losses

# ============================================================
# 3. Example Usage
# ============================================================
#if __name__ == "__main__":
if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # === Create the OR-dataset (same logic as before) ===
    def or_generator(X_k):
        # X_k: (seq_len=10, dim=5)
        return np.logical_or(X_k[-1], X_k[-2]).astype(float)
    #
    n_train, n_val, n_test = 8000, 1000, 1000
    seq_len, dim = 10, 5
    #
    def make_data(n):
        X = np.random.randint(0, 2, (n, seq_len, dim)).astype(float)
        y = np.array([or_generator(X[k]) for k in range(n)], dtype=float)
        return X, y
    #
    X_train, y_train = make_data(n_train)
    X_val, y_val = make_data(n_val)
    X_test, y_test = make_data(n_test)
    #
    # === DataLoaders ===
    batch_size = 64
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True)
    #
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=batch_size)
    #
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    ), batch_size=batch_size)
    #
    # === Initialize and Train ===
    input_dim = seq_len * dim  # flattened input
    model = MLP(input_dim=input_dim, hidden_dims=[128, 64], output_dim=dim, dropout=0.1)
    #
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=100, lr=1e-3, patience=10, device=device
    )
    

if 1:
    # === Plot Losses ===
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('MLP Training vs Validation Loss')
    plt.show()

    # === Test Evaluation ===
if 1:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
            preds.append(outputs > 0.5)
            trues.append(y_batch.numpy())
    #
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = (preds == trues).mean()
    print(f"\nTest Accuracy: {acc:.4f}")  # 1.000



####################### sequence to scalar


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Transformer Model: sequence → scalar
# ============================================================
class TransformerSequenceToScalar(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1,
                 output_type="regression"):
        """
        output_type: "regression" or "binary"
        """
        super().__init__()
        self.model_dim = model_dim
        self.output_type = output_type
        #
        self.input_fc = nn.Linear(input_dim, model_dim)
        #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #
        # take last token embedding for prediction
        self.output_fc = nn.Linear(model_dim, 1)
        #self.output_fc = nn.Linear(model_dim * k, 1)  # <-- update in __init__ accordingly
        #
    #
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch,)
        """
        x = self.input_fc(x)
        x = self.transformer(x)
        k = 3
        last_k = x[:, -k:, :].reshape(x.size(0), -1)  # flatten last k embeddings → (batch, k * model_dim)
        self.output_fc = nn.Linear(64 * k, 1)  # <-- update in __init__ accordingly
        out = self.output_fc(last_k).squeeze(-1)
        #last_token = x[:, -1, :]          # take last token's embedding
        #out = self.output_fc(last_token).squeeze(-1)
        if self.output_type == "binary":
            out = torch.sigmoid(out)
        return out


# ============================================================
# 2. Synthetic Dataset Generator
# ============================================================
def make_dataset(n_samples=2000, seq_len=10, dim=5, task="binary"):
    """
    Generate synthetic sequences X and labels y.
    - For binary classification: y = 1 if mean(X) > 0 else 0
    - For regression: y = mean(X)
    """
    X = np.random.randn(n_samples, seq_len, dim)
    if task == "binary":
        y = (X.mean(axis=(1, 2)) > 0).astype(float)
    else:  # regression
        y = X.mean(axis=(1, 2))
    return X, y


# ============================================================
# 3. Training Function
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device="cpu", task="binary"):
    if task == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    train_losses, val_losses = [], []
    #
    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        #
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # ---------- VALIDATION ----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        #
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    #
    return train_losses, val_losses


# ============================================================
# 4. Main Script
# ============================================================
#if __name__ == "__main__":
if 1:
    # ---- choose task: "binary" or "regression"
    task = "binary"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # ---- make datasets
    X_train, y_train = make_dataset(5000, seq_len=10, dim=5, task=task)
    X_val,   y_val   = make_dataset(1000, seq_len=10, dim=5, task=task)
    X_test,  y_test  = make_dataset(1000, seq_len=10, dim=5, task=task)
    #
    # ---- convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    #
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)
    #
    # ---- data loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    #
    # ---- model
    model = TransformerSequenceToScalar(input_dim=5, model_dim=64, num_heads=4,
                                        num_layers=2, dropout=0.1, output_type=task)
    #
    # ---- training
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           num_epochs=500, lr=1e-3, device=device, task=task)



if 1:
    # ---- plot losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Transformer Training ({task.title()})")
    plt.legend()
    plt.grid(True)
    plt.savefig('sequence_to_scalar_v2.png')
    plt.show()

if 1:
    # ---- test evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy()
            preds.append(out)
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    #
    if task == "binary":
        preds_binary = (preds > 0.5).astype(float)
        acc = (preds_binary == trues).mean()
        print(f"\nTest Accuracy: {acc:.4f}")  ### 0.9790
    else:
        mse = np.mean((preds - trues)**2)
        print(f"\nTest MSE: {mse:.4f}")


############ the same with aba -> 1, else 0

# ============================================================
# 2. Synthetic Dataset Generator
# ============================================================

dim = 5
a = np.random.randn(1, dim)
b = np.random.randn(1, dim)

def make_dataset(n_samples=2000, seq_len=10, dim=5, task="binary"):
    """
    Generate synthetic sequences X and labels y.
    - For binary classification: y = 1 if mean(X) > 0 else 0
    - For regression: y = mean(X)
    """
    X = np.random.randn(n_samples, seq_len, dim)
    y = np.random.rand(n_samples)
    for i in range(0, n_samples):
        # select 3 random locations in 0..seq_len-1
        loc = np.random.choice(np.arange(0, seq_len), 3, replace=False)
        X[i, loc[0]] = a
        X[i, loc[1]] = b
        X[i, loc[2]] = a
        if (np.min([loc[0], loc[2]]) < loc[1]) & (loc[1] < np.max([loc[0], loc[2]])):
            y[i] = 1.0
        else:
            y[i] = 0.0
    #  
    return X, y


# ============================================================
# 3. Training Function
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device="cpu", task="binary"):
    if task == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    train_losses, val_losses = [], []
    #
    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        #
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # ---------- VALIDATION ----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        #
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    #
    return train_losses, val_losses


# ============================================================
# 4. Main Script
# ============================================================
#if __name__ == "__main__":
if 1:
    # ---- choose task: "binary" or "regression"
    task = "binary"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # ---- make datasets
    X_train, y_train = make_dataset(4000, seq_len=10, dim=5, task=task)
    X_val,   y_val   = make_dataset(1000, seq_len=10, dim=5, task=task)
    X_test,  y_test  = make_dataset(1000, seq_len=10, dim=5, task=task)
    #
    # ---- convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    #
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)
    #
    # ---- data loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    #
    # ---- model
    model = TransformerSequenceToScalar(input_dim=5, model_dim=64, num_heads=4,
                                        num_layers=2, dropout=0.1, output_type=task)
    #
    # ---- training
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           num_epochs=1000, lr=1e-3, device=device, task=task)



if 1:
    # ---- plot losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Transformer Training ({task.title()})")
    plt.legend()
    plt.grid(True)
    plt.savefig('sequence_to_scalar_v2.png')
    plt.show()

if 1:
    # ---- test evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy()
            preds.append(out)
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    #
    if task == "binary":
        preds_binary = (preds > 0.5).astype(float)
        acc = (preds_binary == trues).mean()
        print(f"\nTest Accuracy: {acc:.4f}")  ### 0.45
    else:
        mse = np.mean((preds - trues)**2)
        print(f"\nTest MSE: {mse:.4f}")

#################################
###### other approach, based on letter sequences:

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import string
import random
import matplotlib.pyplot as plt


# ============================================================
# 1. Synthetic dataset: letter sequences with rule-based labels
# ============================================================
def contains_aba(seq: str) -> int:
    """Return 1 if 'b' is between two 'a's, else 0."""
    for i in range(1, len(seq) - 1):
        #if seq[i] == 'b' and seq[i - 1] == 'a' and seq[i + 1] == 'a':
        #if seq[i] < 'o' and seq[i - 1] == 'f' and seq[i + 1] > 'f':
        #if seq[i] <= 'z' and seq[i - 1] == 'f' and seq[i + 1] >= 'a':
        if seq[i] <= 'k' and 'k'< seq[i - 1] and seq[i-1] < 'r' and 'r' <= seq[i + 1]:
            return 1
    return 0

def generate_dataset(n_samples=5000, seq_len=8):
    alphabet = list(string.ascii_lowercase)
    sequences, labels = [], []
    for _ in range(n_samples):
        seq = ''.join(random.choices(alphabet, k=seq_len))
        sequences.append(seq)
        labels.append(contains_aba(seq))
    return sequences, np.array(labels, dtype=float)


# ============================================================
# 2. Character encoding utilities
# ============================================================
class CharTokenizer:
    def __init__(self):
        self.vocab = list(string.ascii_lowercase)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    #
    def encode(self, seq):
        return [self.char2idx[c] for c in seq]
    #
    def encode_batch(self, seqs):
        return torch.tensor([self.encode(s) for s in seqs], dtype=torch.long)


# ============================================================
# 3. Transformer Model for sequence → scalar
# ============================================================
class TransformerLetterClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, model_dim=64,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_fc = nn.Linear(embed_dim, model_dim)
        #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, 1)
    #
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = self.input_fc(x)            # (batch, seq_len, model_dim)
        x = self.transformer(x)         # (batch, seq_len, model_dim)
        pooled = x.mean(dim=1)          # average across sequence positions
        out = self.output_fc(pooled).squeeze(-1)
        return torch.sigmoid(out)       # binary classification (0–1)


# ============================================================
# 4. Training and Evaluation
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    #
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        #
        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        #
        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    return train_losses, val_losses


# ============================================================
# 5. Main
# ============================================================
#if __name__ == "__main__":
if 1:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    #
    seq_len = 8
    train_seqs, train_y = generate_dataset(8000, seq_len)
    val_seqs,   val_y   = generate_dataset(1000, seq_len)
    test_seqs,  test_y  = generate_dataset(1000, seq_len)
    #
    tokenizer = CharTokenizer()
    #
    X_train = tokenizer.encode_batch(train_seqs)
    X_val   = tokenizer.encode_batch(val_seqs)
    X_test  = tokenizer.encode_batch(test_seqs)
    #
    y_train = torch.tensor(train_y, dtype=torch.float32)
    y_val   = torch.tensor(val_y, dtype=torch.float32)
    y_test  = torch.tensor(test_y, dtype=torch.float32)
    #
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    model = TransformerLetterClassifier(vocab_size=tokenizer.vocab_size)
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           num_epochs=250, lr=1e-3, device=device)


if 1:
    # plot losses
    plt.figure(figsize=(7,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.title("Training vs Validation Loss")
    plt.savefig("letter_sequences_aba_v3.png")
    plt.show()

if 1:
    # evaluate on test set
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            preds.append(outputs)
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = ((preds > 0.5).astype(float) == trues).mean()
    print(f"\nTest Accuracy: {acc:.4f}")  
    ### 0.8850, np.sum(train_y)/len(train_y) == 0.0925
    ### 0.9150, np.sum(train_y)/len(train_y) == 0.2068
    ### 0.7190,  np.sum(train_y)/len(train_y) == 0.1975

example (case 2):
preds[33:38]
array([1.1092579e-11, 1.8448012e-11, 9.7479558e-01, 1.1126758e-11,
       1.7761188e-11], dtype=float32)
train_seqs[33:38]
['vspzravh', 'ryddcohp', 'sfqgmxvc', 'lhauqgto', 'labwxovp']