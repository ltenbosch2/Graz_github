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


