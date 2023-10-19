import torch
import matplotlib.pyplot as plt


"""
    Function to generate Sine Wave Dataset
    Generate Sine Waves based on random Phase, Frequency and Amplitude
    
    Args:
        samples (int): Number of data samples. Defaults to 1000.
        length (int): Number of data length per sample. Defaults to 100.
        overlapping (int): Number of overlapping sine waves. Defaults to 3.
        amplitude (list): Min and Max amplitude of data. Defaults to [0.1, 1].
        frequency (list): Min and Max frequency of data. Defaults to [0.001, 1].
        phase (list): Min and Max phase of data. Defaults to [-m.pi, m.pi].
        add_noise (bool): Gaussian Noise is added or not. Defaults to True.

    Returns:
        data (torch.tensor): Sine Wave Dataset of Shape: samples x 1 x length.

"""
def get_sine_wave_dataset(samples: int = 10000,
                          length: int = 100,
                          overlapping: int = 3,
                          amplitude: list = [0.1, 1],
                          frequency: list = [0.005, 0.2],
                          phase: list = [0, 2*3.141],
                          add_noise: bool = True) -> torch.tensor:

    x = torch.arange(0, length, 1).repeat(samples, 1)
    data = 0
    for i in range(overlapping):
        a = (torch.rand(samples) *
             (amplitude[1]-amplitude[0]) + amplitude[0]).unsqueeze(-1)
        f = (torch.rand(samples) *
             (frequency[1]-frequency[0]) + frequency[0]).unsqueeze(-1)
        p = (torch.rand(samples) *
             (phase[1]-phase[0]) + phase[0]).unsqueeze(-1)
        data = data + (a*torch.sin(f*x+p)).unsqueeze(1)

    # add noise
    if add_noise:
        mean = 0
        std = 0.05
        data = data + torch.randn(data.shape) * std + mean
    return data


if __name__ == '__main__':
    dataset = get_sine_wave_dataset()
    print("Dataset shape:", dataset.shape)

    # plot one example
    fig = plt.figure()
    plt.plot(dataset[0, 0])
    plt.show()
