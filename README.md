# Deep learning · Postgraduate hands-on

Train, inspect, and experiment with neural networks in a **90-minute hands-on session**.
Start with image classification, move to sequences, then generate new images with a VAE.

## Core notebooks

Open a notebook on GitHub to read it, or click **Open in Colab** to run it in your browser.

| Notebook | What you will do | Time | Run |
|---|---|---:|---|
| [1 · Multilayer perceptrons](01_mlp.ipynb) | Classify clothing images. Inspect errors and experiment with regularisation. | 20 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/01_mlp.ipynb) |
| [2 · CNNs & transfer learning](02_cnn.ipynb) | Compare a CNN with the MLP. Use ResNet-18 on pet breeds and inspect saliency and Grad-CAM. | 22 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/02_cnn.ipynb) |
| [3 · Recurrent networks](03_rnn.ipynb) | Classify IMDB movie reviews with word embeddings and a GRU. Compare scores and training time. | 22 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/03_rnn.ipynb) |
| [4 · Autoencoders & VAEs](04_vae.ipynb) | Learn from colored shapes. Inspect reconstructions, sample new images, and interpolate in latent space. | 20 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/04_vae.ipynb) |

The activities total 84 minutes, leaving a few minutes for transitions.
If time runs short, move notebook 4 to homework.

## Optional demos

| Notebook | What you will explore | Run |
|---|---|---|
| [0 · Universal approximation](00_uat.ipynb) | Watch a network learn a 2D decision boundary, then compare hidden-layer widths. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/00_uat.ipynb) |
| [5 · Melody generation](05_mus.ipynb) | Train a GRU on Nottingham melodies and listen to continuations at different temperatures. Playback uses a fixed duration per note. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Macuyiko/dl-course-notebooks/blob/main/Postgraduate/05_mus.ipynb) |

## Getting started

1. Click **Open in Colab** next to a notebook.
2. For faster training, select **Runtime → Change runtime type** and choose a GPU if available.
3. Select **Runtime → Run all**, or work through the cells in order.
4. Change the settings in the exercises and rerun the relevant cells to compare results.

Use `QUICK = True` for the in-class run and `QUICK = False` for longer experiments
where that setting is available. A GPU is useful for longer GRU runs; the shapes VAE also runs on CPU.

### Data and setup

| Notebook | Data | Download |
|---|---|---|
| 0 · Universal approximation | Generated 2D examples | None |
| 1 · MLP | Fashion-MNIST | About 30 MB |
| 2 · CNN | Fashion-MNIST + Oxford-IIIT Pets | About 800 MB for Pets, plus pretrained weights |
| 3 · RNN | IMDB movie reviews | About 80 MB |
| 4 · VAE | Generated shapes in three colors | None |
| 5 · Melody generation | Nottingham MIDI files | Downloaded automatically |

Run the download cells before class. Downloaded data is cached within the runtime;
starting a fresh Colab runtime requires downloading it again.

Colab provides the main dependencies, including PyTorch, torchvision, NumPy,
matplotlib, scikit-learn, and IPython. Notebook 3 installs `datasets` if needed;
notebook 5 installs `pretty_midi` if needed. No Kaggle account or API key is required.

## Shared training loop

The same five steps recur throughout the notebooks:

```python
optimizer.zero_grad()
out = model(xb)
loss = criterion(out, yb)
loss.backward()
optimizer.step()
```

The model and loss change. The training loop stays the same.

## Reference results

Recorded results from `QUICK = True` CPU runs are included below for comparison.
Results can vary with settings, package versions, and random seeds.

<details>
<summary>Show classification scores and observations</summary>


| Model | Parameters | Task | Train | Validation | Test |
|---|---:|---|---:|---:|---:|
| MLP | 0.20 M | Fashion-MNIST, 28×28 | 0.92 | 0.84 | 0.852 |
| CNN | 0.06 M | Fashion-MNIST, 28×28 | 0.89 | 0.86 | 0.863 |
| Frozen ResNet-18 + linear head | 11.2 M | 37 pet breeds | 0.96 | 0.93 | 0.812 |
| Bag of embeddings | 0.64 M | IMDB sentiment | 0.87 | 0.83 | 0.818 |
| GRU | 0.67 M | IMDB sentiment | 0.89 | 0.81 | 0.820 |

AUC is reported with accuracy for the image models: macro AUC is 0.987 for the MLP and 0.988 for the CNN.
The programme's earlier tabular session introduces AUC.

### What to look for

- The MLP reaches 0.852 test accuracy on 28×28 Fashion-MNIST. Chance level is 0.10.
- The CNN reaches 0.863 with fewer parameters than the MLP.
- The confusion matrix shows common errors between similar clothing classes, such as Shirt and T-shirt.
- The pretrained ResNet reaches 0.812 on 37 pet breeds.
- The GRU reaches similar test accuracy to bag of embeddings: 0.820 versus 0.818. It takes longer to train.

</details>
