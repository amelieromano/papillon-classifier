# Feature Plan: Papillon Classifier

## What I'm building
A classifier that looks at a dog photo and predicts: is this a Papillon or not?

## How it will work
1. Load 32 Papillon photos and 31 other dog photos
2. Use a pre-trained AI model (ResNet18) and fine-tune it on our photos
3. Test it on a new photo and see if it guesses correctly

## Tools
- Python
- PyTorch
- ResNet18 (transfer learning)

## Success = 
The model correctly identifies a Papillon photo it has never seen before
