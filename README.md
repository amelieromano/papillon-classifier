---
title: Papillon Classifier
emoji: 🦋
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# Is it a Papillon? 🦋

Upload a photo of any dog and this little classifier will tell you whether it's a Papillon — those ridiculously fluffy butterfly-eared spaniels — or just a regular (still very good) dog.

## What it does

Drop in a dog photo and hit the button. The model runs a prediction and shows you a result styled like a zine stamp: **PAPILLON!** if it thinks yes, or a quieter "not a papillon" if not. Confidence percentage included. The illustration switches between a calm sitting pose (no) and a full sprint (yes).

## How I built it

This was my first machine learning project, so I kept things focused.

I collected about 63 dog photos — 32 Papillons and 31 other breeds — and used **transfer learning** with a pre-trained **ResNet18** from torchvision. I replaced the final fully-connected layer with a 2-class output (papillon / not papillon), then fine-tuned the whole network for 10 epochs using Adam with a 0.0001 learning rate. The small dataset size is why transfer learning made sense here: the base ResNet18 already knows what a dog looks like, so it just needed to learn what makes a Papillon different.

Training ran on CPU in a few minutes. Final accuracy on a held-out test set: **91.7% (11/12 correct)**.

The UI is a Gradio app styled to match my portfolio's zine-meets-puppy-Pinterest aesthetic — dark brown background, rust and ochre accents, Fredoka One headings, Caveat for the handwritten labels, offset box shadows. The three dog illustrations (sitting, running, portrait) are custom SVGs drawn to match the palette.

## What I learned

- Transfer learning is genuinely powerful for small datasets. Training a ResNet18 from scratch on 63 images would have been useless.
- Data quantity matters more than I expected — 63 images is not a lot, and the model still has obvious failure modes with unusual lighting or partial views.
- Gradio makes deploying a model extremely easy. The hardest part of putting this on Hugging Face Spaces was getting Git LFS set up for the `.pth` file.
- I now have strong opinions about butterfly ears.

## Tech stack

- Python · PyTorch · torchvision · Gradio · Pillow
- ResNet18 (pretrained on ImageNet, fine-tuned on my dataset)
- Deployed on Hugging Face Spaces
