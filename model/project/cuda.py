import torch
import torch.nn as nn
import torch.optim as optim
import time

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Простая модель для теста
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.fc(x)

# Инициализация модели и данных
model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Генерация данных
batch_size = 256
input_data = torch.randn(batch_size, 1024).to(device)
target_data = torch.randn(batch_size, 512).to(device)

# Тест производительности
print("Starting benchmark...")
start_time = time.time()

for step in range(100000):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Benchmark completed in {end_time - start_time:.2f} seconds")

# Проверка загрузки GPU
print("Run `nvidia-smi` in your terminal to check GPU usage.")