from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Dropout, Module, Dropout2d, Flatten

class RotationModel(Module):
    def __init__(self, image_zise=(112, 96)):
        super().__init__()
        self.model = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Dropout2d(0.25),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Dropout2d(0.5),

            Flatten(),
            Linear(256 * (image_zise[0]//16) * (image_zise[1] // 16), 1024),
            ReLU(),
            Dropout(0.5),
            Linear(1024, 1)  # 输出1个角度
        )

    def forward(self, x):
        return self.model(x)